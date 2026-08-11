from __future__ import annotations

from pathlib import Path

import torch

from gtcrn_coreai.loader import (
    ATTENTION_STATE_SHAPE,
    CONVOLUTION_STATE_SHAPE,
    INTER_FRAME_STATE_SHAPE,
    create_streaming_model,
    initial_state,
)
from gtcrn_coreai.model import GTCRN

ROOT = Path(__file__).resolve().parents[1]


def test_initial_state_shapes_and_dtype() -> None:
    state = initial_state(dtype=torch.float16)
    assert tuple(state[0].shape) == CONVOLUTION_STATE_SHAPE
    assert tuple(state[1].shape) == ATTENTION_STATE_SHAPE
    assert tuple(state[2].shape) == INTER_FRAME_STATE_SHAPE
    assert all(value.dtype == torch.float16 for value in state)
    assert all(torch.count_nonzero(value) == 0 for value in state)


def test_functional_wrapper_does_not_mutate_inputs() -> None:
    model = create_streaming_model(ROOT / "weights/model_trained_on_dns3.tar")
    spectrogram = torch.randn(1, 257, 1, 2)
    state = initial_state()
    before = tuple(value.clone() for value in state)

    with torch.no_grad():
        outputs = model(spectrogram, *state)

    assert tuple(outputs[0].shape) == (1, 257, 1, 2)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(state, before, strict=True)
    )
    assert any(torch.count_nonzero(value) > 0 for value in outputs[1:])


def test_streaming_frames_match_offline_model() -> None:
    checkpoint_path = ROOT / "weights/model_trained_on_dns3.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    offline_model = GTCRN().eval()
    offline_model.load_state_dict(checkpoint["model"], strict=True)
    streaming_model = create_streaming_model(checkpoint_path)

    spectrogram = torch.randn(1, 257, 12, 2)
    with torch.no_grad():
        expected = offline_model(spectrogram)
        state = initial_state()
        frames = []
        for index in range(spectrogram.shape[2]):
            outputs = streaming_model(spectrogram[:, :, index : index + 1], *state)
            frames.append(outputs[0])
            state = outputs[1:]
        actual = torch.cat(frames, dim=2)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
