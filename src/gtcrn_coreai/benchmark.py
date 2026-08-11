"""Benchmark streaming GTCRN across Core AI compute-unit specializations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Compute-unit preferences are supplied by the OS runtime.
os.environ.setdefault("USE_OS_COREAI", "1")

from .audio import spectrogram_frames  # noqa: E402
from .interface import INPUT_NAMES, OUTPUT_NAMES  # noqa: E402
from .loader import initial_state  # noqa: E402

FRAME_PERIOD_MS = 16.0


def specialization_options(compute_unit: str) -> Any:
    """Build explicit Core AI specialization options for a benchmark run."""

    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    constructors = {
        "cpu": SpecializationOptions.cpu_only,
        "gpu": lambda: SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.gpu()
        ),
        "neural-engine": lambda: SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        ),
        "default": SpecializationOptions.default,
    }
    return constructors[compute_unit]()


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Return stable latency metrics for one consecutive-frame run."""

    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "realtime_ratio_mean": float(values.mean() / FRAME_PERIOD_MS),
    }


async def benchmark(
    asset_path: Path,
    wav_path: Path,
    *,
    compute_unit: str,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, Any]:
    """Measure end-to-end inference-call latency with recurrent state feedback."""

    from coreai.runtime import AIModel, NDArray

    total_frames = warmup_frames + measured_frames
    frames = spectrogram_frames(wav_path, total_frames).to(torch.float16)
    if frames.shape[2] < total_frames:
        raise ValueError(
            f"Requested {total_frames} frames, but {wav_path} has {frames.shape[2]}"
        )
    frame_inputs = [
        NDArray(data=frames[:, :, index : index + 1].contiguous())
        for index in range(total_frames)
    ]
    state = tuple(
        NDArray(data=value.contiguous())
        for value in initial_state(dtype=torch.float16)
    )

    options = specialization_options(compute_unit)
    load_start = time.perf_counter_ns()
    aimodel = await AIModel.load(asset_path, options)
    load_ms = (time.perf_counter_ns() - load_start) / 1e6
    function = aimodel.load_function("main")
    if tuple(function.desc.input_names) != INPUT_NAMES:
        raise AssertionError(f"Unexpected input names: {function.desc.input_names}")
    if tuple(function.desc.output_names) != OUTPUT_NAMES:
        raise AssertionError(f"Unexpected output names: {function.desc.output_names}")

    latencies_ms: list[float] = []
    for index, frame in enumerate(frame_inputs):
        inputs = dict(zip(INPUT_NAMES, (frame, *state), strict=True))
        start = time.perf_counter_ns()
        outputs = await function(inputs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        state = tuple(outputs[name] for name in OUTPUT_NAMES[1:])
        if index >= warmup_frames:
            latencies_ms.append(elapsed_ms)

    result: dict[str, Any] = {
        "asset": str(asset_path),
        "compute_unit": compute_unit,
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "frame_period_ms": FRAME_PERIOD_MS,
        "model_load_ms": load_ms,
        **summarize_latencies(latencies_ms),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset",
        type=Path,
        nargs="?",
        default=Path("exports/gtcrn_dns3_float16_streaming.aimodel"),
    )
    parser.add_argument("--wav", type=Path, default=Path("test_wavs/mix.wav"))
    parser.add_argument(
        "--compute-unit",
        choices=("cpu", "gpu", "neural-engine", "default"),
        required=True,
    )
    parser.add_argument("--warmup-frames", type=int, default=64)
    parser.add_argument("--frames", type=int, default=512)
    args = parser.parse_args()
    result = asyncio.run(
        benchmark(
            args.asset,
            args.wav,
            compute_unit=args.compute_unit,
            warmup_frames=args.warmup_frames,
            measured_frames=args.frames,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
