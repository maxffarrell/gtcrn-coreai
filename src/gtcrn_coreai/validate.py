"""Validate an exported Core AI model against its PyTorch source graph."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Delegate specialization is supplied by the OS Core AI framework. Selecting it
# before importing the runtime lets the validator use CPU-only execution for a
# stable correctness comparison, independent of accelerator placement.
os.environ.setdefault("USE_OS_COREAI", "1")

from coreai.runtime import (  # noqa: E402
    AIModel,
    ComputeUnitKind,
    NDArray,
    SpecializationOptions,
)

from .export import INPUT_NAMES, OUTPUT_NAMES
from .loader import create_streaming_model, initial_state


def psnr(reference: np.ndarray, actual: np.ndarray) -> float:
    """Return peak signal-to-noise ratio in decibels."""

    error = np.mean(np.square(reference.astype(np.float64) - actual.astype(np.float64)))
    if error == 0:
        return float("inf")
    peak = max(float(np.max(np.abs(reference))), 1e-12)
    return 10.0 * np.log10((peak * peak) / error)


def spectrogram_frames(wav_path: Path, frame_count: int) -> torch.Tensor:
    """Load real 16 kHz audio and return GTCRN-compatible STFT frames."""

    samples, sample_rate = sf.read(wav_path, dtype="float32")
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")
    if samples.ndim != 1:
        samples = samples.mean(axis=-1)
    waveform = torch.from_numpy(samples)
    window = torch.hann_window(512).sqrt()
    spec = torch.stft(
        waveform,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=window,
        return_complex=True,
    )
    return torch.view_as_real(spec).unsqueeze(0)[:, :, :frame_count]


async def validate(
    asset_path: Path,
    checkpoint_path: Path,
    wav_path: Path,
    *,
    frame_count: int,
    dtype: torch.dtype,
    compute_unit: str,
) -> list[float]:
    """Run consecutive real frames through PyTorch and the Core AI runtime."""

    frames = spectrogram_frames(wav_path, frame_count).to(dtype)
    source_model = create_streaming_model(checkpoint_path, dtype=dtype)
    torch_state = initial_state(dtype=dtype)
    coreai_state = tuple(value.clone() for value in torch_state)

    specialization_options = {
        "cpu": SpecializationOptions.cpu_only,
        "gpu": lambda: SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.gpu()
        ),
        "default": SpecializationOptions.default,
    }[compute_unit]()
    aimodel = await AIModel.load(asset_path, specialization_options)
    function = aimodel.load_function("main")
    if tuple(function.desc.input_names) != INPUT_NAMES:
        raise AssertionError(f"Unexpected input names: {function.desc.input_names}")
    if tuple(function.desc.output_names) != OUTPUT_NAMES:
        raise AssertionError(f"Unexpected output names: {function.desc.output_names}")

    scores: list[float] = []
    for index in range(frames.shape[2]):
        frame = frames[:, :, index : index + 1].contiguous()
        with torch.no_grad():
            torch_outputs = source_model(frame, *torch_state)
        torch_state = tuple(value.detach() for value in torch_outputs[1:])

        named_inputs = dict(
            zip(
                INPUT_NAMES,
                (frame, *coreai_state),
                strict=True,
            )
        )
        runtime_outputs = await function(
            {
                name: NDArray(data=value.contiguous())
                for name, value in named_inputs.items()
            }
        )
        ordered = tuple(runtime_outputs[name].numpy() for name in OUTPUT_NAMES)
        coreai_state = tuple(torch.from_numpy(value.copy()) for value in ordered[1:])
        scores.append(psnr(torch_outputs[0].detach().cpu().numpy(), ordered[0]))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--variant", choices=("dns3", "vctk"), default="dns3")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--wav", type=Path, default=Path("test_wavs/mix.wav"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--compute-unit",
        choices=("cpu", "gpu", "default"),
        default="cpu",
        help="Core AI specialization used for parity (default: cpu).",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint or Path("weights") / f"model_trained_on_{args.variant}.tar"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    scores = asyncio.run(
        validate(
            args.asset,
            checkpoint,
            args.wav,
            frame_count=args.frames,
            dtype=dtype,
            compute_unit=args.compute_unit,
        )
    )
    print(f"Compute unit: {args.compute_unit}")
    print(f"Frames: {len(scores)}")
    print(f"Minimum PSNR: {min(scores):.2f} dB")
    print(f"Mean PSNR: {np.mean(scores):.2f} dB")
    if min(scores) < 40.0:
        raise SystemExit("Core AI parity failed: PSNR below 40 dB")


if __name__ == "__main__":
    main()
