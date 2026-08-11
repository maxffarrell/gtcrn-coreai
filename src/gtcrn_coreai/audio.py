"""Audio feature helpers shared by validation and benchmarking."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


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
