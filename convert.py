#!/usr/bin/env python3
"""Convert and palettize the pinned SigLIP SO400M vision tower."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel
from transformers import __version__ as transformers_version

MODEL_ID = "google/siglip-so400m-patch14-384"
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
INPUT_SIZE = 384
OUTPUT_DIM = 1152
RELEASE_VERSION = "0.2.0"
VARIANTS = {
    "p8": {
        "bits": 8,
        "filename": "SigLIPSO400M384-P8.mlpackage",
        "fp16_filename": "SigLIPSO400M384-FP16.mlpackage",
        "minimum_deployment_target": "iOS17",
        "granularity": "per_tensor",
        "group_size": None,
        "quantization": "palette-kmeans-8bit-per-tensor",
        "compression": "kmeans-8bit-scalar-per-tensor",
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p8-kmeans8pt-v1"
        ),
    },
    "p6": {
        "bits": 6,
        "filename": "SigLIPSO400M384-P6.mlpackage",
        "fp16_filename": "SigLIPSO400M384-FP16.mlpackage",
        "minimum_deployment_target": "iOS17",
        "granularity": "per_tensor",
        "group_size": None,
        "quantization": "palette-kmeans-6bit-per-tensor",
        "compression": "kmeans-6bit-scalar-per-tensor",
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6-kmeans6pt-v1"
        ),
    },
    "p6g16": {
        "bits": 6,
        "filename": "SigLIPSO400M384-P6G16.mlpackage",
        "fp16_filename": "SigLIPSO400M384-iOS18-FP16.mlpackage",
        "minimum_deployment_target": "iOS18",
        "granularity": "per_grouped_channel",
        "group_size": 16,
        "quantization": "palette-kmeans-6bit-grouped-channel-16",
        "compression": "kmeans-6bit-scalar-per-grouped-channel-16",
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6g16-kmeans6gc16-v1"
        ),
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_model(model: ct.models.MLModel, output: Path) -> None:
    staging = output.with_name(f".{output.stem}.tmp-{os.getpid()}{output.suffix}")
    previous = output.with_name(f".{output.stem}.previous{output.suffix}")
    for stale in (staging, previous):
        if stale.exists():
            shutil.rmtree(stale)
    model.save(str(staging))
    if output.exists():
        output.replace(previous)
    try:
        staging.replace(output)
    except Exception:
        if previous.exists() and not output.exists():
            previous.replace(output)
        raise
    if previous.exists():
        shutil.rmtree(previous)


class SiglipEmbedding(torch.nn.Module):
    def __init__(self, tower: SiglipVisionModel) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.tower(pixel_values=pixel_values, return_dict=False)[1]
        denominator = torch.sqrt(torch.sum(pooled * pooled, dim=-1, keepdim=True))
        return pooled / torch.clamp(denominator, min=1e-6)


def load_torch_model() -> SiglipEmbedding:
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    tower = (
        SiglipVisionModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .cpu()
        .eval()
    )
    tower.config._attn_implementation = "eager"
    tower.config.output_attentions = False
    tower.config.output_hidden_states = False
    if (
        int(tower.config.image_size) != INPUT_SIZE
        or int(tower.config.hidden_size) != OUTPUT_DIM
    ):
        raise ValueError(
            "The pinned checkpoint no longer has the expected vision shape"
        )
    return SiglipEmbedding(tower).cpu().eval()


def set_common_metadata(model: ct.models.MLModel, minimum_deployment_target: str) -> None:
    model.author = "miracle2k"
    model.license = "Apache-2.0; derived from google/siglip-so400m-patch14-384"
    model.version = RELEASE_VERSION
    model.short_description = (
        "SigLIP SO400M 384px vision tower; L2-normalized 1152-vector"
    )
    model.input_description["image"] = "384x384 RGB image"
    model.output_description["embedding"] = (
        "L2-normalized Float32 embedding, shape 1x1152"
    )
    model.user_defined_metadata.update(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "input_transform": "RGB uint8 -> x / 127.5 - 1",
            "output_transform": "L2 normalization",
            "embedding_dimension": str(OUTPUT_DIM),
            "minimum_deployment_target": minimum_deployment_target,
            "release_repository": "https://github.com/miracle2k/siglip-so400m-coreml",
            "release_version": RELEASE_VERSION,
        }
    )


def convert_fp16(
    output: Path, trace_path: Path, minimum_deployment_target: str
) -> dict[str, Any]:
    wrapped = load_torch_model()
    example = torch.linspace(
        -1.0,
        1.0,
        steps=3 * INPUT_SIZE * INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
    started = time.perf_counter()
    with torch.inference_mode():
        eager = wrapped(example)
        traced = torch.jit.trace(
            wrapped, example, strict=True, check_trace=False
        ).eval()
        traced_result = traced(example)
    torch.testing.assert_close(traced_result, eager, rtol=1e-4, atol=1e-5)
    graph = str(traced.inlined_graph)
    forbidden = [
        name
        for name in ("_native_multi_head_attention", "scaled_dot_product_attention")
        if name in graph
    ]
    if forbidden:
        raise RuntimeError(
            f"Trace contains unsupported attention operations: {forbidden}"
        )
    traced.save(str(trace_path))
    trace_seconds = time.perf_counter() - started

    started = time.perf_counter()
    try:
        deployment_target = getattr(ct.target, minimum_deployment_target)
    except AttributeError as error:
        raise ValueError(
            f"Unsupported Core ML deployment target: {minimum_deployment_target}"
        ) from error
    model = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                color_layout=ct.colorlayout.RGB,
                scale=1.0 / 127.5,
                bias=[-1.0, -1.0, -1.0],
            )
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        minimum_deployment_target=deployment_target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    set_common_metadata(model, minimum_deployment_target)
    model.user_defined_metadata["quantization"] = "fp16"
    save_model(model, output)
    del wrapped, traced, model
    gc.collect()
    return {
        "conversion_seconds": time.perf_counter() - started,
        "minimum_deployment_target": minimum_deployment_target,
        "package_bytes": package_size(output),
        "package_tree_sha256": package_tree_sha256(output),
        "trace_seconds": trace_seconds,
    }


def compress(fp16_path: Path, output: Path, variant: str) -> dict[str, Any]:
    config = VARIANTS[variant]
    started = time.perf_counter()
    baseline = ct.models.MLModel(str(fp16_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    palette_options: dict[str, Any] = {
        "mode": "kmeans",
        "nbits": int(config["bits"]),
        "granularity": str(config["granularity"]),
        "cluster_dim": 1,
        "enable_per_channel_scale": False,
        "num_kmeans_workers": min(8, os.cpu_count() or 1),
        "weight_threshold": 2048,
    }
    if config["group_size"] is not None:
        palette_options["group_size"] = int(config["group_size"])
    palette = cto.coreml.OpPalettizerConfig(
        **palette_options,
    )
    compressed = cto.coreml.palettize_weights(
        baseline,
        config=cto.coreml.OptimizationConfig(global_config=palette),
    )
    set_common_metadata(compressed, str(config["minimum_deployment_target"]))
    compressed.user_defined_metadata.update(
        {
            "embedding_space_id": str(config["embedding_space_id"]),
            "quantization": str(config["quantization"]),
        }
    )
    save_model(compressed, output)
    result = {
        "compression": str(config["compression"]),
        "compression_seconds": time.perf_counter() - started,
        "minimum_deployment_target": config["minimum_deployment_target"],
        "package_bytes": package_size(output),
        "package_tree_sha256": package_tree_sha256(output),
    }
    del compressed, baseline
    gc.collect()
    return result


def synthetic_image() -> Image.Image:
    y, x = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
    pixels = np.stack(
        ((13 * x + 7 * y) % 256, (3 * x + 17 * y) % 256, (19 * x + 5 * y) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def smoke_test(path: Path) -> dict[str, Any]:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    output = np.asarray(model.predict({"image": synthetic_image()})["embedding"])
    if output.shape != (1, OUTPUT_DIM) or output.dtype != np.float32:
        raise ValueError(
            f"Invalid output contract from {path.name}: {output.shape}, {output.dtype}"
        )
    vector = output[0]
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite output from {path.name}")
    norm = float(np.linalg.norm(vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"Output from {path.name} is not normalized: {norm}")
    return {"finite": True, "norm": norm, "shape": list(output.shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build"))
    parser.add_argument(
        "--variants", choices=tuple(VARIANTS), nargs="+", default=("p8", "p6")
    )
    parser.add_argument("--reuse-fp16", action="store_true")
    parser.add_argument("--worker", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.worker:
        variant = args.worker
        fp16_path = output / str(VARIANTS[variant]["fp16_filename"])
        result = compress(
            fp16_path, output / str(VARIANTS[variant]["filename"]), variant
        )
        write_json(output / f"worker-{variant}.json", result)
        return

    report: dict[str, Any] = {
        "environment": {
            "coremltools": ct.__version__,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "variants": {},
    }
    fp16_paths: dict[str, Path] = {}
    for variant in args.variants:
        config = VARIANTS[variant]
        target = str(config["minimum_deployment_target"])
        fp16_path = output / str(config["fp16_filename"])
        existing = fp16_paths.setdefault(target, fp16_path)
        if existing != fp16_path:
            raise ValueError(f"Conflicting FP16 bases for {target}")

    fp16_results: dict[str, dict[str, Any]] = {}
    fp16_smoke_tests: dict[str, dict[str, Any]] = {}
    for target, fp16_path in fp16_paths.items():
        if not args.reuse_fp16:
            trace_name = (
                "SigLIPSO400M384-traced.pt"
                if target == "iOS17"
                else f"SigLIPSO400M384-{target}-traced.pt"
            )
            fp16_results[target] = convert_fp16(
                fp16_path, output / trace_name, target
            )
        elif not fp16_path.exists():
            raise FileNotFoundError(f"Missing FP16 intermediate: {fp16_path.name}")
        fp16_smoke_tests[target] = smoke_test(fp16_path)

    if len(fp16_paths) == 1:
        only_target = next(iter(fp16_paths))
        if not args.reuse_fp16:
            report["fp16"] = fp16_results[only_target]
        report["fp16_smoke_test"] = fp16_smoke_tests[only_target]
    else:
        if not args.reuse_fp16:
            report["fp16"] = fp16_results
        report["fp16_smoke_test"] = fp16_smoke_tests
    write_json(output / "build-report.json", report)

    for variant in args.variants:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--worker",
                variant,
            ],
            check=True,
        )
        worker_path = output / f"worker-{variant}.json"
        variant_result = json.loads(worker_path.read_text())
        worker_path.unlink()
        variant_path = output / str(VARIANTS[variant]["filename"])
        variant_result["smoke_test"] = smoke_test(variant_path)
        report["variants"][variant] = variant_result
        write_json(output / "build-report.json", report)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
