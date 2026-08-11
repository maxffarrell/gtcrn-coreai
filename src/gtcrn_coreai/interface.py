"""Stable names for the exported streaming GTCRN interface."""

INPUT_NAMES = (
    "spectrogram",
    "convolution_state",
    "attention_state",
    "inter_frame_state",
)
OUTPUT_NAMES = (
    "enhanced_spectrogram",
    "convolution_state_out",
    "attention_state_out",
    "inter_frame_state_out",
)
