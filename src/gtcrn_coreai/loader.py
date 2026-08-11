"""Load the attributed GTCRN checkpoints into the streaming architecture."""

from __future__ import annotations

from pathlib import Path

import torch

from .convert import convert_to_stream
from .model import GTCRN
from .stream import StreamGTCRN

CONVOLUTION_STATE_SHAPE = (2, 1, 16, 16, 33)
ATTENTION_STATE_SHAPE = (2, 3, 1, 1, 16)
INTER_FRAME_STATE_SHAPE = (2, 1, 33, 16)


class FunctionalStreamGTCRN(torch.nn.Module):
    """Expose GTCRN caches as functional I/O for reliable Core AI export.

    The upstream streaming implementation updates cache slices in place. Cloning
    at the graph boundary preserves the same math while preventing export-time
    alias analysis from treating cache inputs as dead values.
    """

    def __init__(self, model: StreamGTCRN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        spectrogram: torch.Tensor,
        convolution_state: torch.Tensor,
        attention_state: torch.Tensor,
        inter_frame_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(
            spectrogram,
            convolution_state.clone(),
            attention_state.clone(),
            inter_frame_state.clone(),
        )


def initial_state(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-filled state tensors for the first audio frame."""

    return (
        torch.zeros(CONVOLUTION_STATE_SHAPE, dtype=dtype, device=device),
        torch.zeros(ATTENTION_STATE_SHAPE, dtype=dtype, device=device),
        torch.zeros(INTER_FRAME_STATE_SHAPE, dtype=dtype, device=device),
    )


def create_streaming_model(
    checkpoint_path: Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> FunctionalStreamGTCRN:
    """Load an upstream checkpoint and return an export-safe streaming model."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_model = GTCRN().eval()
    source_model.load_state_dict(checkpoint["model"], strict=True)

    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, source_model)
    stream_model.to(dtype=dtype)
    return FunctionalStreamGTCRN(stream_model).eval()
