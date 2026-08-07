#!/usr/bin/env python3
"""Load released Core ML packages and verify their public contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
from PIL import Image

MODEL_ID = "google/siglip-so400m-patch14-384"
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
INPUT_SIZE = 384
OUTPUT_DIM = 1152
VARIANT_BY_QUANTIZATION = {
    "palette-kmeans-8bit-per-tensor": {
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p8-kmeans8pt-v1"
        ),
        "minimum_deployment_target": "iOS17",
        "variant": "p8",
    },
    "palette-kmeans-6bit-per-tensor": {
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6-kmeans6pt-v1"
        ),
        "minimum_deployment_target": "iOS17",
        "variant": "p6",
    },
    "palette-kmeans-6bit-grouped-channel-16": {
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6g16-kmeans6gc16-v1"
        ),
        "minimum_deployment_target": "iOS18",
        "variant": "p6g16",
    },
}


def synthetic_image() -> Image.Image:
    y, x = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
    pixels = np.stack(
        ((13 * x + 7 * y) % 256, (3 * x + 17 * y) % 256, (19 * x + 5 * y) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def verify(path: Path) -> tuple[str, np.ndarray, dict[str, Any]]:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    metadata = model.user_defined_metadata
    if (
        metadata.get("model_id") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError(f"Unexpected model provenance in {path.name}")
    config = VARIANT_BY_QUANTIZATION.get(metadata.get("quantization", ""))
    if config is None:
        raise ValueError(f"Unexpected quantization in {path.name}")
    if metadata.get("embedding_space_id") != config["embedding_space_id"]:
        raise ValueError(f"Unexpected embedding space in {path.name}")
    if metadata.get("minimum_deployment_target") != config["minimum_deployment_target"]:
        raise ValueError(f"Unexpected deployment target in {path.name}")
    output = np.asarray(model.predict({"image": synthetic_image()})["embedding"])
    if output.shape != (1, OUTPUT_DIM) or output.dtype != np.float32:
        raise ValueError(
            f"Invalid output contract from {path.name}: {output.shape}, {output.dtype}"
        )
    vector = output[0]
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite output from {path.name}")
    variant = str(config["variant"])
    norm = float(np.linalg.norm(vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"Output from {path.name} is not normalized: {norm}")
    return (
        variant,
        vector,
        {
            "embedding_space_id": config["embedding_space_id"],
            "minimum_deployment_target": config["minimum_deployment_target"],
            "compute_units": "CPU_ONLY",
            "finite": True,
            "model": path.name,
            "norm": norm,
            "shape": list(output.shape),
            "variant": variant,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", type=Path, nargs="+")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parent / "tests/reference_outputs.npz",
    )
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()

    vectors: dict[str, np.ndarray] = {}
    reports: list[dict[str, Any]] = []
    for supplied in args.packages:
        variant, vector, report = verify(supplied.expanduser().resolve())
        if variant in vectors:
            raise ValueError(f"Duplicate {variant} package")
        vectors[variant] = vector
        reports.append(report)

    reference_path = args.reference.expanduser().resolve()
    if args.write_reference:
        stored_vectors: dict[str, np.ndarray] = {}
        if reference_path.exists():
            with np.load(reference_path, allow_pickle=False) as reference:
                if str(reference["model_id"].item()) != MODEL_ID:
                    raise ValueError("Reference model ID does not match")
                if str(reference["model_revision"].item()) != MODEL_REVISION:
                    raise ValueError("Reference model revision does not match")
                stored_vectors = {
                    name: np.asarray(reference[name], dtype=np.float32)
                    for name in reference.files
                    if name not in {"model_id", "model_revision"}
                }
        stored_vectors.update(vectors)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference_path,
            model_id=np.asarray(MODEL_ID),
            model_revision=np.asarray(MODEL_REVISION),
            **stored_vectors,
        )
    else:
        with np.load(reference_path, allow_pickle=False) as reference:
            if str(reference["model_id"].item()) != MODEL_ID:
                raise ValueError("Reference model ID does not match")
            if str(reference["model_revision"].item()) != MODEL_REVISION:
                raise ValueError("Reference model revision does not match")
            for report in reports:
                variant = str(report["variant"])
                expected = np.asarray(reference[variant], dtype=np.float32).reshape(-1)
                actual = vectors[variant]
                cosine = float(
                    np.dot(actual, expected)
                    / (np.linalg.norm(actual) * np.linalg.norm(expected))
                )
                maximum_error = float(np.max(np.abs(actual - expected)))
                if cosine < 0.9999 or maximum_error > 0.005:
                    raise ValueError(
                        f"{variant} differs from the release reference: cosine={cosine}, max error={maximum_error}"
                    )
                report["reference_cosine"] = cosine
                report["reference_maximum_absolute_error"] = maximum_error

    print(
        json.dumps(
            {"models": reports, "reference": reference_path.name},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
