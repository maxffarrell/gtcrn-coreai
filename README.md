# GTCRN for Apple Core AI

A reproducible, streaming [Core AI](https://developer.apple.com/documentation/coreai)
conversion of [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), the 48.2K-parameter
speech-enhancement network introduced in *GTCRN: A Speech Enhancement Model
Requiring Ultralow Computational Resources*.

This repository contains the conversion source, both MIT-licensed upstream
checkpoints, and ready-to-load float16 `.aimodel` assets. It targets Apple's new
Core AI framework—not the older Core ML `.mlmodel`/`.mlpackage` format.

> [!IMPORTANT]
> This is an independent conversion, not an official release from the GTCRN
> authors or Apple. The model graph and bundled weights are attributed to Rong
> Xiaobin and contributors; see [Attribution and license](#attribution-and-license).

## What is included

| Artifact | Purpose |
| --- | --- |
| `exports/gtcrn_dns3_float16_streaming.aimodel` | Ready-to-use DNS3 checkpoint |
| `exports/gtcrn_vctk_float16_streaming.aimodel` | Ready-to-use VCTK-DEMAND checkpoint |
| `weights/*.tar` | Unmodified upstream PyTorch checkpoints |
| `src/gtcrn_coreai/export.py` | Reproducible PyTorch → Core AI conversion |
| `src/gtcrn_coreai/validate.py` | Consecutive-frame PyTorch/Core AI parity check |
| `src/gtcrn_coreai/benchmark.py` | Stateful per-frame specialization benchmark |

GTCRN expects a complex short-time Fourier transform (STFT), not raw audio. The
export processes one frame every 256 samples: **16 ms at 16 kHz**. Its recurrent
and causal-convolution state is explicit, which makes cross-call behavior
deterministic and avoids relying on runtime state that may reset between calls.

## Quick start

You need an Apple-silicon Mac and Python 3.11–3.13. [`uv`](https://docs.astral.sh/uv/)
is recommended because the lockfile pins the working toolchain.

```sh
brew install uv
uv sync
```

Export either pretrained variant:

```sh
uv run gtcrn-coreai-export --variant dns3 --overwrite
uv run gtcrn-coreai-export --variant vctk --overwrite
```

This runs the same high-level path used by Apple's
[`coreai-models`](https://github.com/apple/coreai-models) recipes:

1. Load the upstream PyTorch checkpoint and convert it to the causal streaming
   architecture.
2. Export a static-shape `torch.export.ExportedProgram`.
3. Apply Core AI's decomposition table.
4. Convert with `TorchConverter`, optimize, attach provenance metadata, and save
   a self-contained `.aimodel` asset.

The default is float16, Apple's recommended baseline for efficient on-device
inference. This network is only about 48K parameters, so aggressive weight
compression offers little absolute size benefit and is intentionally not
applied without an audio-quality study.

## Model interface

All tensors are float16 in the bundled assets.

| Name | Direction | Static shape | Meaning |
| --- | --- | --- | --- |
| `spectrogram` | input | `[1, 257, 1, 2]` | One complex STFT frame as real/imaginary pairs |
| `convolution_state` | input | `[2, 1, 16, 16, 33]` | Encoder/decoder causal convolution state |
| `attention_state` | input | `[2, 3, 1, 1, 16]` | Encoder/decoder temporal-attention GRU state |
| `inter_frame_state` | input | `[2, 1, 33, 16]` | Dual-path inter-frame GRU state |
| `enhanced_spectrogram` | output | `[1, 257, 1, 2]` | Enhanced complex STFT frame |
| `convolution_state_out` | output | `[2, 1, 16, 16, 33]` | State for the next frame |
| `attention_state_out` | output | `[2, 3, 1, 1, 16]` | State for the next frame |
| `inter_frame_state_out` | output | `[2, 1, 33, 16]` | State for the next frame |

Initialize all three state inputs to zero at the start of a stream. After each
inference call, feed each `*_out` value into its corresponding state input. Use
the original settings for reconstruction:

- mono audio at 16 kHz
- 512-point FFT and 512-sample window
- 256-sample hop
- square root of a Hann window

The model asset intentionally keeps STFT/iSTFT and overlap-add outside the
neural graph. In an app, those operations belong in the real-time audio pipeline.

## Python runtime validation

On a Core AI-capable macOS installation, compare consecutive real audio frames
against PyTorch:

```sh
uv run gtcrn-coreai-validate \
  exports/gtcrn_dns3_float16_streaming.aimodel \
  --variant dns3 \
  --frames 8
```

The command verifies function names and shapes, carries state across calls, and
requires at least 40 dB PSNR for every enhanced spectral frame. It defaults to
CPU-only specialization to isolate numerical correctness from accelerator
placement; pass `--compute-unit gpu` to validate GPU specialization too. This is
a graph parity check; perceptual quality should also be evaluated on a
representative speech/noise corpus before shipping a product.

Run the source checks with:

```sh
uv run ruff check .
uv run pytest
```

Benchmark a specialization over consecutive frames with recurrent state
feedback (timing excludes STFT and model load):

```sh
uv run gtcrn-coreai-benchmark --compute-unit cpu
uv run gtcrn-coreai-benchmark --compute-unit gpu
```

See [Neural Engine evaluation](docs/ane-evaluation.md) for the ANE experiment,
failure evidence, and the decision to ship the current graph with CPU/GPU
specialization rather than claim partial or unstable ANE acceleration.

### Validation results

Measured on an Apple M4 Max running macOS 27.0 with `coreai-core 1.0.0b2` and
`coreai-torch 0.4.1`. Each Core AI result carries all three state tensors across
eight consecutive frames from the included real test recording.

| Asset | Specialization | Minimum PSNR | Mean PSNR |
| --- | --- | ---: | ---: |
| DNS3 float16 | CPU | 56.16 dB | 72.23 dB |
| DNS3 float16 | GPU | 60.44 dB | 71.33 dB |
| VCTK float16 | CPU | 62.36 dB | 71.37 dB |
| VCTK float16 | GPU | 64.93 dB | 72.57 dB |

The streaming PyTorch graph was also compared with the original full-context
causal graph across all 611 frames of `test_wavs/mix.wav`: maximum absolute
spectral error was `1.05e-5` and mean absolute error was `1.74e-8`.

> [!NOTE]
> Neural Engine residency is not claimed. Explicit/default ANE specialization
> cannot load the current decomposed PReLU/GRU graph on the validation machine,
> so there is no valid ANE latency result to compare. CPU and GPU specialization
> both execute and pass parity. The measured decision and reproduction details
> are in [docs/ane-evaluation.md](docs/ane-evaluation.md).

## Ahead-of-time compilation

The `.aimodel` assets can be loaded directly. For ahead-of-time compilation and
deployment diagnostics, install the full current Xcode release and use Apple's
`coreai-build` tool:

```sh
xcrun coreai-build compile \
  exports/gtcrn_dns3_float16_streaming.aimodel \
  --platform iOS
```

Compiler availability, flags, and supported deployment versions follow the
installed Xcode release. See Apple's
[ahead-of-time compilation documentation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)
for the current platform options. A successful Python conversion does not by
itself prove Neural Engine residency or real-time performance on every device;
profile the compiled asset on the devices you intend to support. Ahead-of-time
`coreai-build` compilation was not run for this release because the validation
machine had Command Line Tools rather than full Xcode installed.

## Design notes

- **Faithful weights and math.** The architecture and checkpoint mapping come
  from the official GTCRN implementation. No retraining or unmeasured
  quantization is performed.
- **Functional state boundary.** The upstream streaming model updates cache
  slices in place. The export wrapper clones state tensors at its boundary so
  `torch.export` retains them as live inputs and produces explicit state outputs.
- **Static one-frame input.** A fixed shape is appropriate for low-latency audio
  and gives Core AI the best opportunity to specialize the graph.
- **Float16 baseline.** It balances size, precision, and Apple-silicon execution.
  Float32 remains available with `--dtype float32` for debugging.

## Attribution and license

The original GTCRN repository, architecture, streaming implementation, test
audio, and checkpoints are Copyright © 2024 Rong Xiaobin and are distributed
under the MIT License. This repository preserves that license and source
attribution. The Core AI conversion code is Copyright © 2026 Max Farrell and is
also MIT licensed.

Please cite the original paper when using the model:

> Xiaobin Rong et al., “GTCRN: A Speech Enhancement Model Requiring Ultralow
> Computational Resources,” ICASSP 2024, DOI
> [10.1109/ICASSP48485.2024.10448310](https://doi.org/10.1109/ICASSP48485.2024.10448310).

Full provenance and third-party details are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); machine-readable citation
metadata is in [CITATION.cff](CITATION.cff).
