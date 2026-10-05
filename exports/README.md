# Core AI assets

> **Original GTCRN model and weights: Xiaobin Rong and the GTCRN authors/contributors.** [Official upstream](https://github.com/Xiaobin-Rong/gtcrn) · [Original paper](https://doi.org/10.1109/ICASSP48485.2024.10448310) · [Full attribution](../ATTRIBUTION.md). Max Farrell provides the Core AI conversion and validation. Upstream MIT terms are retained.

The checked-in `.aimodel` directories are self-contained Core AI AIPrograms.
Their parameter data comes from the attributed upstream checkpoints in
[`weights/`](../weights/).

Regenerate them from source:

```sh
uv run gtcrn-coreai-export --variant dns3 --overwrite
uv run gtcrn-coreai-export --variant vctk --overwrite
```

The assets are float16 and use a single static STFT frame per invocation. State
is explicit functional input/output, so callers must feed each `*_out` tensor
back into the matching input on the next frame.

| Asset | `main.mlirb` SHA-256 |
| --- | --- |
| DNS3 | `3c8c21435d1f42fc18dcf027c418f5e55112b43b3514362eda654ff71bcdd581` |
| VCTK | `2882ee3046862f7403c279053e603cb619583e0dc79646acace297740c147f22` |

CPU and GPU specializations pass the parity validator. Neural Engine residency
has not been established; see the validation status in the root README.
