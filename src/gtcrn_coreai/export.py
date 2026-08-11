"""Convert pretrained streaming GTCRN checkpoints to Core AI AIPrograms."""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

from .interface import INPUT_NAMES, OUTPUT_NAMES
from .loader import create_streaming_model, initial_state

UPSTREAM_SOURCE_TIMESTAMP = 1_785_748_511


def _metadata(variant: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "Rong Xiaobin et al.; Core AI conversion by Max Farrell"
    metadata.license = "MIT"
    metadata.model_description = (
        "Streaming GTCRN speech enhancement model trained on "
        f"{variant.upper()}. Original implementation and weights: "
        "https://github.com/Xiaobin-Rong/gtcrn. Converted with Apple's "
        "coreai-torch reference workflow."
    )
    metadata.creation_date = int(
        os.environ.get("SOURCE_DATE_EPOCH", UPSTREAM_SOURCE_TIMESTAMP)
    )
    return metadata


def export_model(
    checkpoint: Path,
    output: Path,
    *,
    variant: str,
    dtype: torch.dtype,
    overwrite: bool = False,
) -> Path:
    """Export one checkpoint to a self-contained ``.aimodel`` asset."""

    if output.suffix != ".aimodel":
        raise ValueError("Output path must end in .aimodel")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    model = create_streaming_model(checkpoint, dtype=dtype)
    example_inputs = (
        torch.randn(1, 257, 1, 2, dtype=dtype),
        *initial_state(dtype=dtype),
    )

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The tensor attributes .*_flat_weights.* were assigned during export\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        exported = torch.export.export(model, args=example_inputs, strict=False)
        exported = exported.run_decompositions(get_decomp_table())

    program = (
        TorchConverter()
        .add_exported_program(
            exported_program=exported,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
        )
        .to_coreai()
    )
    program.optimize()
    output.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(output, _metadata(variant))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("dns3", "vctk"),
        default="dns3",
        help="Upstream pretrained checkpoint to convert (default: dns3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: weights/model_trained_on_<variant>.tar).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .aimodel path (default: exports/gtcrn_<variant>_fp16_streaming.aimodel).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Export precision (default: float16).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint or root / "weights" / f"model_trained_on_{args.variant}.tar"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output = (
        args.output
        or root / "exports" / f"gtcrn_{args.variant}_{args.dtype}_streaming.aimodel"
    )
    result = export_model(
        checkpoint,
        output,
        variant=args.variant,
        dtype=dtype,
        overwrite=args.overwrite,
    )
    print(f"Saved Core AI model to {result}")


if __name__ == "__main__":
    main()
