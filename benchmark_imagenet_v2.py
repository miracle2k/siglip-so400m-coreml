#!/usr/bin/env python3
"""Measure Core ML conversion fidelity with a pinned ImageNet-v2 protocol.

The benchmark intentionally reads class names and prompts from a pinned
checkout of google-research/big_vision instead of carrying a second copy.
Images are cached after the original 438px bilinear resize and 384px center
crop so every candidate receives exactly the same uint8 pixels.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import string
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel

MODEL_ID = "google/siglip-so400m-patch14-384"
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
BIG_VISION_COMMIT = "430df79cd16dd6f7f882fd8c0ed2e7c9ea4c1db8"
DATASET_NAME = "imagenet_v2/matched-frequency:3.0.0"
DATASET_ARCHIVE_BYTES = 1_264_079_360
DATASET_ARCHIVE_SHA256 = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
PUBLISHED_TOP1 = 0.772
IMAGE_COUNT = 10_000
CLASS_COUNT = 1_000
PROMPT_COUNT = 81
INPUT_SIZE = 384
RESIZE_SIZE = 438
OUTPUT_DIM = 1_152


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_package(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def literal_assignment(path: Path, variable: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{variable} in {path} is not a string list")
            return value
    raise ValueError(f"Could not find {variable} in {path}")


def canonicalize(text: str, keep: str | None = None) -> str:
    text = text.replace("_", " ")
    if keep:
        text = keep.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_protocol(big_vision_root: Path) -> dict[str, Any]:
    root = big_vision_root.expanduser().resolve()
    class_path = root / "big_vision/datasets/imagenet/class_names.py"
    prompt_path = (
        root
        / "big_vision/evaluators/proj/image_text/prompt_engineering_constants.py"
    )
    raw_classes = literal_assignment(class_path, "CLIP_IMAGENET_CLASS_NAMES")
    raw_prompts = literal_assignment(prompt_path, "CLIP_PAPER_PROMPT_TEMPLATES")
    classes = [canonicalize(name, keep=",").split(",")[0] for name in raw_classes]
    prompts = [canonicalize(template, keep="{}") for template in raw_prompts]
    if len(classes) != CLASS_COUNT or len(prompts) != PROMPT_COUNT:
        raise ValueError(
            f"Unexpected protocol dimensions: {len(classes)} classes, {len(prompts)} prompts"
        )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != BIG_VISION_COMMIT:
        raise ValueError(f"Expected Big Vision {BIG_VISION_COMMIT}, found {commit}")
    return {
        "big_vision_commit": commit,
        "class_names": classes,
        "class_names_sha256": sha256_json(classes),
        "prompt_templates": prompts,
        "prompt_templates_sha256": sha256_json(prompts),
        "source_files": {
            "class_names_sha256": sha256_file(class_path),
            "prompt_templates_sha256": sha256_file(prompt_path),
        },
    }


def image_paths(dataset_root: Path) -> tuple[list[Path], np.ndarray]:
    root = dataset_root.expanduser().resolve()
    nested = root / "imagenetv2-matched-frequency-format-val"
    if nested.is_dir():
        root = nested
    paths: list[Path] = []
    labels: list[int] = []
    for label in range(CLASS_COUNT):
        directory = root / str(label)
        if not directory.is_dir():
            raise ValueError(f"Missing ImageNet-v2 class directory: {directory}")
        members = sorted(
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if len(members) != 10:
            raise ValueError(f"Expected 10 images in {directory}, found {len(members)}")
        paths.extend(members)
        labels.extend([label] * len(members))
    if len(paths) != IMAGE_COUNT:
        raise ValueError(f"Expected {IMAGE_COUNT} images, found {len(paths)}")
    return paths, np.asarray(labels, dtype=np.int16)


def resize_for_big_vision(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(torch.float32)
    resized = F.interpolate(
        tensor,
        size=(RESIZE_SIZE, RESIZE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    resized = resized.clamp_(0, 255).to(torch.uint8)
    offset = (RESIZE_SIZE - INPUT_SIZE) // 2
    cropped = resized[0, :, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    return cropped.permute(1, 2, 0).contiguous().numpy()


def prepare_image_cache(dataset_root: Path, work_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    cache_path = work_dir / "imagenet_v2_images_uint8.npy"
    labels_path = work_dir / "imagenet_v2_labels.npy"
    metadata_path = work_dir / "imagenet_v2_cache.json"
    paths, labels = image_paths(dataset_root)
    manifest = [[str(path.relative_to(dataset_root)), path.stat().st_size] for path in paths]
    expected = {
        "count": IMAGE_COUNT,
        "dataset": DATASET_NAME,
        "manifest_sha256": sha256_json(manifest),
        "preprocessing": "decode-rgb-resize438-bilinear-noaa-uint8-centercrop384-v1",
        "shape": [IMAGE_COUNT, INPUT_SIZE, INPUT_SIZE, 3],
    }
    if cache_path.exists() and labels_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        cached_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if existing == expected and cached.shape == tuple(expected["shape"]) and np.array_equal(cached_labels, labels):
            return cache_path, labels_path, expected
        raise ValueError("Existing ImageNet-v2 cache does not match this protocol")

    temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=tuple(expected["shape"])
    )
    started = time.monotonic()
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            cache[index] = resize_for_big_vision(image)
        if (index + 1) % 250 == 0:
            elapsed = time.monotonic() - started
            eta = elapsed / (index + 1) * (len(paths) - index - 1)
            print(f"preprocess {index + 1}/{len(paths)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    cache.flush()
    del cache
    temporary.replace(cache_path)
    np.save(labels_path, labels, allow_pickle=False)
    write_json(metadata_path, expected)
    return cache_path, labels_path, expected


def torch_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def text_prototypes(
    protocol: dict[str, Any], work_dir: Path, device: torch.device, batch_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    output_path = work_dir / "imagenet_v2_text_prototypes.npz"
    expected_metadata = {
        "big_vision_commit": protocol["big_vision_commit"],
        "class_names_sha256": protocol["class_names_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_templates_sha256": protocol["prompt_templates_sha256"],
        "protocol": "big-vision-clip-paper-prompts-first-alias-v1",
    }
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            prototypes = np.asarray(data["prototypes"], dtype=np.float32)
        if metadata != expected_metadata or prototypes.shape != (CLASS_COUNT, OUTPUT_DIM):
            raise ValueError("Existing text prototypes do not match this protocol")
        return prototypes, metadata

    prompts = [
        template.format(class_name)
        for class_name in protocol["class_names"]
        for template in protocol["prompt_templates"]
    ]
    prompt_labels = np.repeat(np.arange(CLASS_COUNT, dtype=np.int32), PROMPT_COUNT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = SiglipTextModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().to(device)
    sums = np.zeros((CLASS_COUNT, OUTPUT_DIM), dtype=np.float64)
    counts = np.zeros(CLASS_COUNT, dtype=np.int32)
    started = time.monotonic()
    with torch.inference_mode():
        for start, prompt_batch in batches(prompts, batch_size):
            encoded = tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_attention_mask=False,
                return_tensors="pt",
            )
            vectors = model(input_ids=encoded["input_ids"].to(device), return_dict=False)[1]
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors_np = vectors.to(torch.float32).cpu().numpy()
            labels = prompt_labels[start : start + len(prompt_batch)]
            np.add.at(sums, labels, vectors_np)
            np.add.at(counts, labels, 1)
            completed = start + len(prompt_batch)
            if completed % (batch_size * 20) == 0 or completed == len(prompts):
                elapsed = time.monotonic() - started
                eta = elapsed / completed * (len(prompts) - completed)
                print(f"text {completed}/{len(prompts)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    if not np.array_equal(counts, np.full(CLASS_COUNT, PROMPT_COUNT)):
        raise RuntimeError("Text prototype prompt counts are incomplete")
    prototypes = (sums / counts[:, None]).astype(np.float32)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(expected_metadata, sort_keys=True)),
            prototypes=prototypes,
        )
    temporary.replace(output_path)
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return prototypes, expected_metadata


def percentile(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def classification_metrics(
    embeddings: np.ndarray, labels: np.ndarray, prototypes: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors / np.maximum(norms[:, None], 1e-12)
    predictions = np.argmax(vectors @ prototypes.T, axis=1).astype(np.int16)
    correct = predictions == labels
    per_class = np.bincount(labels[correct], minlength=CLASS_COUNT) / np.bincount(
        labels, minlength=CLASS_COUNT
    )
    return (
        {
            "accuracy_top1": float(np.mean(correct)),
            "correct": int(np.sum(correct)),
            "count": len(labels),
            "macro_class_accuracy_top1": float(np.mean(per_class)),
            "output_norm": percentile(norms),
            "published_top1": PUBLISHED_TOP1,
            "percentage_point_delta_vs_published": float((np.mean(correct) - PUBLISHED_TOP1) * 100),
        },
        predictions,
        correct,
    )


def save_run_arrays(
    work_dir: Path,
    variant: str,
    predictions: np.ndarray,
    correct: np.ndarray,
    durations: np.ndarray,
) -> None:
    output = work_dir / f"{variant}-predictions.npz"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            correct=np.asarray(correct, dtype=np.bool_),
            prediction=np.asarray(predictions, dtype=np.int16),
            prediction_seconds=np.asarray(durations, dtype=np.float64),
        )
    temporary.replace(output)


def evaluate_coreml(
    variant: str,
    package: Path,
    image_cache: Path,
    labels: np.ndarray,
    prototypes: np.ndarray,
    work_dir: Path,
    compute_units: ct.ComputeUnit,
) -> dict[str, Any]:
    package = package.expanduser().resolve()
    package_digest = sha256_package(package)
    embeddings_path = work_dir / f"{variant}-embeddings.npy"
    progress_path = work_dir / f"{variant}-progress.json"
    result_path = work_dir / f"{variant}.json"
    expected_progress = {
        "completed": 0,
        "package_sha256": package_digest,
        "variant": variant,
    }
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["package_sha256"] != package_digest:
            raise ValueError(f"Existing {variant} result uses a different package")
        return result

    images = np.load(image_cache, mmap_mode="r", allow_pickle=False)
    if embeddings_path.exists():
        progress = json.loads(progress_path.read_text())
        if {key: progress[key] for key in expected_progress if key != "completed"} != {
            key: expected_progress[key] for key in expected_progress if key != "completed"
        }:
            raise ValueError(f"Existing {variant} progress does not match the package")
        completed = int(progress["completed"])
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        durations_path = work_dir / f"{variant}-durations.npy"
        durations = np.lib.format.open_memmap(durations_path, mode="r+")
    else:
        completed = 0
        embeddings = np.lib.format.open_memmap(
            embeddings_path, mode="w+", dtype=np.float32, shape=(len(images), OUTPUT_DIM)
        )
        durations_path = work_dir / f"{variant}-durations.npy"
        durations = np.lib.format.open_memmap(
            durations_path, mode="w+", dtype=np.float64, shape=(len(images),)
        )
        write_json(progress_path, expected_progress)

    load_started = time.monotonic()
    model = ct.models.MLModel(str(package), compute_units=compute_units)
    load_seconds = time.monotonic() - load_started
    started = time.monotonic()
    for index in range(completed, len(images)):
        image = Image.fromarray(np.asarray(images[index]), mode="RGB")
        prediction_started = time.perf_counter()
        output = model.predict({"image": image})["embedding"]
        durations[index] = time.perf_counter() - prediction_started
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape != (OUTPUT_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid {variant} output at image {index}: {vector.shape}")
        embeddings[index] = vector
        if (index + 1) % 100 == 0 or index + 1 == len(images):
            embeddings.flush()
            durations.flush()
            write_json(progress_path, {**expected_progress, "completed": index + 1})
        if (index + 1) % 250 == 0:
            elapsed = time.monotonic() - started
            processed = index + 1 - completed
            eta = elapsed / max(processed, 1) * (len(images) - index - 1)
            print(f"{variant} {index + 1}/{len(images)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    metrics, predictions, correct = classification_metrics(embeddings, labels, prototypes)
    result = {
        "compute_units": str(compute_units.name),
        "load_seconds": load_seconds,
        "metrics": metrics,
        "model": package.name,
        "package_bytes": sum(item.stat().st_size for item in package.rglob("*") if item.is_file()),
        "package_sha256": package_digest,
        "prediction_latency_ms": {
            key: value * 1000 for key, value in percentile(np.asarray(durations)).items()
        },
        "variant": variant,
    }
    save_run_arrays(work_dir, variant, predictions, correct, np.asarray(durations))
    write_json(result_path, result)
    progress_path.unlink(missing_ok=True)
    return result


def evaluate_pytorch(
    image_cache: Path,
    labels: np.ndarray,
    prototypes: np.ndarray,
    work_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    variant = "pytorch_fp32"
    embeddings_path = work_dir / f"{variant}-embeddings.npy"
    result_path = work_dir / f"{variant}.json"
    durations_path = work_dir / f"{variant}-durations.npy"
    if result_path.exists():
        return json.loads(result_path.read_text())
    images = np.load(image_cache, mmap_mode="r", allow_pickle=False)
    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=(len(images), OUTPUT_DIM)
    )
    durations: list[float] = []
    model = SiglipVisionModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().to(device)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            values = torch.from_numpy(np.asarray(images[start:stop]).copy()).permute(0, 3, 1, 2)
            values = values.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
            batch_started = time.perf_counter()
            vectors = model(pixel_values=values, return_dict=False)[1]
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if device.type == "mps":
                torch.mps.synchronize()
            batch_seconds = time.perf_counter() - batch_started
            embeddings[start:stop] = vectors.cpu().numpy()
            durations.extend([batch_seconds / (stop - start)] * (stop - start))
            if stop % 100 == 0 or stop == len(images):
                embeddings.flush()
            if stop % 250 == 0:
                elapsed = time.monotonic() - started
                eta = elapsed / stop * (len(images) - stop)
                print(f"{variant} {stop}/{len(images)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    durations_array = np.asarray(durations, dtype=np.float64)
    np.save(durations_path, durations_array, allow_pickle=False)
    metrics, predictions, correct = classification_metrics(embeddings, labels, prototypes)
    result = {
        "metrics": metrics,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prediction_latency_ms_per_image": {
            key: value * 1000 for key, value in percentile(durations_array).items()
        },
        "variant": variant,
    }
    save_run_arrays(work_dir, variant, predictions, correct, durations_array)
    write_json(result_path, result)
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def paired_bootstrap_delta(
    candidate_correct: np.ndarray, reference_correct: np.ndarray, iterations: int = 10_000
) -> dict[str, float | int]:
    difference = candidate_correct.astype(np.int8) - reference_correct.astype(np.int8)
    rng = np.random.default_rng(0)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 250):
        count = min(250, iterations - start)
        sample = rng.integers(0, len(difference), size=(count, len(difference)))
        means[start : start + count] = difference[sample].mean(axis=1)
    return {
        "candidate_only_correct": int(np.sum(difference == 1)),
        "mean_accuracy_delta": float(np.mean(difference)),
        "paired_bootstrap_95_high": float(np.quantile(means, 0.975)),
        "paired_bootstrap_95_low": float(np.quantile(means, 0.025)),
        "reference_only_correct": int(np.sum(difference == -1)),
    }


def compare_variants(work_dir: Path, variants: list[str]) -> dict[str, Any]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    embeddings: dict[str, np.ndarray] = {}
    for variant in variants:
        with np.load(work_dir / f"{variant}-predictions.npz", allow_pickle=False) as data:
            arrays[variant] = {key: data[key].copy() for key in ("correct", "prediction")}
        embeddings[variant] = np.load(
            work_dir / f"{variant}-embeddings.npy", mmap_mode="r", allow_pickle=False
        )
    comparisons: dict[str, Any] = {}

    def add_comparison(candidate: str, reference: str) -> None:
        key = f"{candidate}_vs_{reference}"
        reference_vectors = np.asarray(embeddings[reference], dtype=np.float32)
        candidate_vectors = np.asarray(embeddings[candidate], dtype=np.float32)
        reference_vectors = reference_vectors / np.maximum(
            np.linalg.norm(reference_vectors, axis=1, keepdims=True), 1e-12
        )
        candidate_vectors = candidate_vectors / np.maximum(
            np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12
        )
        cosine = np.sum(reference_vectors * candidate_vectors, axis=1)
        comparisons[key] = {
            "embedding_cosine": percentile(cosine),
            "paired_accuracy": paired_bootstrap_delta(
                arrays[candidate]["correct"], arrays[reference]["correct"]
            ),
            "prediction_agreement": float(
                np.mean(arrays[candidate]["prediction"] == arrays[reference]["prediction"])
            ),
        }

    for reference in ("pytorch_fp32", "fp16"):
        if reference not in arrays:
            continue
        for candidate in variants:
            if candidate != reference:
                add_comparison(candidate, reference)

    # P8 is the released deployment baseline, so compare new package candidates
    # against it directly as well as against the FP16 and PyTorch references.
    if "p8" in arrays:
        for candidate in variants:
            if candidate not in {"pytorch_fp32", "fp16", "p8"}:
                add_comparison(candidate, "p8")
    return comparisons


def parse_packages(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Package must use VARIANT=PATH: {value}")
        variant, path = value.split("=", 1)
        if not variant or any(existing == variant for existing, _ in parsed):
            raise ValueError(f"Invalid or duplicate package variant: {variant}")
        parsed.append((variant, Path(path).expanduser().resolve()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--big-vision-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path)
    parser.add_argument("--package", action="append", default=[], metavar="VARIANT=PATH")
    parser.add_argument("--compute-units", choices=("ALL", "CPU_ONLY", "CPU_AND_NE"), default="ALL")
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--torch-batch-size", type=int, default=2)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--skip-pytorch", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    big_vision_root = args.big_vision_root.expanduser().resolve()
    if args.dataset_archive:
        archive = args.dataset_archive.expanduser().resolve()
        if archive.stat().st_size != DATASET_ARCHIVE_BYTES or sha256_file(archive) != DATASET_ARCHIVE_SHA256:
            raise ValueError("ImageNet-v2 archive does not match the pinned TFDS artifact")

    torch.manual_seed(0)
    protocol = load_protocol(big_vision_root)
    image_cache, labels_path, cache_metadata = prepare_image_cache(dataset_root, work_dir)
    labels = np.load(labels_path, allow_pickle=False)
    device = torch_device(args.torch_device)
    prototypes, prototype_metadata = text_prototypes(
        protocol, work_dir, device, args.text_batch_size
    )
    package_values = parse_packages(args.package)
    results: dict[str, Any] = {}
    variants: list[str] = []
    if not args.skip_pytorch:
        results["pytorch_fp32"] = evaluate_pytorch(
            image_cache,
            labels,
            prototypes,
            work_dir,
            device,
            args.torch_batch_size,
        )
        variants.append("pytorch_fp32")

    compute_units = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]
    for variant, package in package_values:
        results[variant] = evaluate_coreml(
            variant,
            package,
            image_cache,
            labels,
            prototypes,
            work_dir,
            compute_units,
        )
        variants.append(variant)

    report = {
        "benchmark": {
            "dataset": DATASET_NAME,
            "dataset_archive_bytes": DATASET_ARCHIVE_BYTES,
            "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            "image_count": IMAGE_COUNT,
            "published_siglip_so400m_384_top1": PUBLISHED_TOP1,
            "protocol": "big-vision-imagenet-v2-zero-shot-v1",
        },
        "cache": cache_metadata,
        "comparisons": compare_variants(work_dir, variants),
        "toolchain": {
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol_sources": {
            "big_vision_commit": BIG_VISION_COMMIT,
            "class_names_sha256": protocol["class_names_sha256"],
            "prompt_templates_sha256": protocol["prompt_templates_sha256"],
            "text_prototypes": prototype_metadata,
        },
        "schema_version": 1,
        "variants": results,
    }
    write_json(work_dir / "imagenet_v2_results.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
