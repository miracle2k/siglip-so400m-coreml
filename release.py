#!/usr/bin/env python3
"""Stamp, validate, and package Core ML release artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import coremltools as ct

MODEL_ID = "google/siglip-so400m-patch14-384"
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
RELEASE_VERSION = "0.2.0"
REPOSITORY = "https://github.com/miracle2k/siglip-so400m-coreml"
VARIANTS = {
    "p8": {
        "bits": 8,
        "compression": "kmeans-8bit-scalar-per-tensor",
        "minimum_deployment_target": "iOS17",
        "model_graph_sha256": "2f64d448265107972b2c41fa7a13f4e436b76c4f819e23d9131f1c7b26b32472",
        "quantization": "palette-kmeans-8bit-per-tensor",
        "weights_sha256": "0e84cef4bcb2fd9c4eee9496809248463998506c7185f57e720a44f874aa04d9",
        "filename": "SigLIPSO400M384-P8.mlpackage",
        "recommended": True,
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p8-kmeans8pt-v1"
        ),
    },
    "p6": {
        "bits": 6,
        "compression": "kmeans-6bit-scalar-per-tensor",
        "minimum_deployment_target": "iOS17",
        "model_graph_sha256": "b888adaf476936ca40eb28b3f27c2b005163d341040d3f415588078a12e11dfd",
        "quantization": "palette-kmeans-6bit-per-tensor",
        "weights_sha256": "d583e30c3583d935ba5b15c3c0adfdbb172779aee40bf63f8973017c0cd194fd",
        "filename": "SigLIPSO400M384-P6.mlpackage",
        "recommended": False,
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6-kmeans6pt-v1"
        ),
    },
    "p6g16": {
        "bits": 6,
        "compression": "kmeans-6bit-scalar-per-grouped-channel-16",
        "minimum_deployment_target": "iOS18",
        "model_graph_sha256": "c9de8d03751a01339314e70d76c6af8b276c21ee578570dbe075f581ca5019a8",
        "quantization": "palette-kmeans-6bit-grouped-channel-16",
        "required_source_package_tree_sha256": "708fd43a1ed2f9ca1d5097c97c5890fe3d5551c4a31c9df1bdedc55826c17b21",
        "weights_sha256": "782c2b3c305b8826151f334a93d0de558a65d705963f138d17ab9c5d540e0843",
        "filename": "SigLIPSO400M384-P6G16.mlpackage",
        "recommended": False,
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6g16-kmeans6gc16-v1"
        ),
    },
}
EXPECTED_PACKAGE_FILES = {
    "Manifest.json",
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
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


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def model_graph_sha256(specification: Any) -> str:
    graph_only = copy.deepcopy(specification)
    graph_only.description.ClearField("metadata")
    return hashlib.sha256(graph_only.SerializeToString(deterministic=True)).hexdigest()


def weight_path(package: Path) -> Path:
    path = package / "Data/com.apple.CoreML/weights/weight.bin"
    if not path.is_file():
        raise ValueError(f"Missing model weights in {package.name}")
    return path


def model_path(package: Path) -> Path:
    path = package / "Data/com.apple.CoreML/model.mlmodel"
    if not path.is_file():
        raise ValueError(f"Missing model specification in {package.name}")
    return path


def validate_package_layout(package: Path) -> None:
    if not package.is_dir():
        raise ValueError(f"Not an mlpackage directory: {package}")
    entries = list(package.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise ValueError(f"Symlinks are not allowed in {package.name}")
    files = {item.relative_to(package).as_posix() for item in entries if item.is_file()}
    if files != EXPECTED_PACKAGE_FILES:
        raise ValueError(
            f"Unexpected files in {package.name}: "
            f"missing={sorted(EXPECTED_PACKAGE_FILES - files)}, "
            f"extra={sorted(files - EXPECTED_PACKAGE_FILES)}"
        )


def validate_source(source: Path, variant: str) -> tuple[Path, Any, str, str, str]:
    source = source.resolve()
    config = VARIANTS[variant]
    validate_package_layout(source)
    specification = ct.utils.load_spec(str(source))
    metadata = specification.description.metadata
    if (
        metadata.userDefined.get("model_id") != MODEL_ID
        or metadata.userDefined.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError(f"Unexpected model provenance in {source.name}")
    if metadata.userDefined.get("quantization") != config["quantization"]:
        raise ValueError(f"Unexpected quantization in {source.name}")
    if (
        metadata.userDefined.get("minimum_deployment_target")
        != config["minimum_deployment_target"]
    ):
        raise ValueError(f"Unexpected deployment target in {source.name}")

    source_tree_hash = tree_sha256(source)
    expected_source_tree_hash = config.get("required_source_package_tree_sha256")
    if expected_source_tree_hash and source_tree_hash != expected_source_tree_hash:
        raise ValueError(f"Unexpected source package tree in {source.name}")
    source_weight_hash = file_sha256(weight_path(source))
    graph_hash = model_graph_sha256(specification)
    if source_weight_hash != config["weights_sha256"]:
        raise ValueError(f"Unexpected weights in {source.name}")
    if graph_hash != config["model_graph_sha256"]:
        raise ValueError(f"Unexpected model graph in {source.name}")
    return source, specification, source_tree_hash, source_weight_hash, graph_hash


def stamp(
    validated: tuple[Path, Any, str, str, str],
    destination: Path,
    variant: str,
    release_version: str,
) -> dict[str, Any]:
    source, specification, source_tree_hash, source_weight_hash, graph_hash = validated
    destination = destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Source and destination package paths must not overlap")
    config = VARIANTS[variant]
    metadata = specification.description.metadata
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    metadata.author = "miracle2k"
    metadata.license = "Apache-2.0; derived from google/siglip-so400m-patch14-384"
    metadata.versionString = release_version
    metadata.userDefined.update(
        {
            "embedding_space_id": str(config["embedding_space_id"]),
            "release_repository": REPOSITORY,
            "release_version": release_version,
        }
    )
    model_path(destination).write_bytes(
        specification.SerializeToString(deterministic=True)
    )
    if shutil.which("xattr"):
        subprocess.run(["xattr", "-cr", str(destination)], check=True)
    for directory in (item for item in destination.rglob("*") if item.is_dir()):
        os.chmod(directory, 0o755)
    for file in (item for item in destination.rglob("*") if item.is_file()):
        os.chmod(file, 0o644)
    os.chmod(destination, 0o755)
    validate_package_layout(destination)
    release_weight_hash = file_sha256(weight_path(destination))
    if release_weight_hash != source_weight_hash:
        raise ValueError(f"Metadata stamping changed the weights in {source.name}")
    return {
        "package_bytes": tree_size(destination),
        "package_tree_sha256": tree_sha256(destination),
        "model_graph_sha256": graph_hash,
        "source_package_tree_sha256": source_tree_hash,
        "weights_sha256": release_weight_hash,
    }


def deterministic_zip(package: Path, output: Path, root: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    entries = [
        (source.relative_to(package.parent).as_posix(), source)
        for source in package.rglob("*")
        if source.is_file()
    ]
    entries.extend((name, root / name) for name in ("LICENSE", "NOTICE"))
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for relative, source in sorted(entries):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with (
                source.open("rb") as reader,
                archive.open(info, "w", force_zip64=True) as writer,
            ):
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    temporary.replace(output)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for variant in VARIANTS:
        parser.add_argument(f"--{variant}", type=Path)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--release-version", default=RELEASE_VERSION)
    parser.add_argument("--write-root-metadata", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dist = args.dist.expanduser().resolve()
    source_paths = {
        variant: supplied.expanduser().resolve()
        for variant in VARIANTS
        if (supplied := getattr(args, variant)) is not None
    }
    if not source_paths:
        parser.error("At least one source package is required")
    if args.write_root_metadata and set(source_paths) != set(VARIANTS):
        parser.error(
            "--write-root-metadata requires a complete P8, P6, and P6G16 release bundle"
        )
    validated = {
        variant: validate_source(source_paths[variant], variant)
        for variant in source_paths
    }
    packages = dist / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    for variant in source_paths:
        config = VARIANTS[variant]
        package = packages / str(config["filename"])
        details = stamp(validated[variant], package, variant, args.release_version)
        archive_name = (
            f"{package.stem}-{config['minimum_deployment_target']}-"
            f"v{args.release_version}.mlpackage.zip"
        )
        archive_path = dist / archive_name
        deterministic_zip(package, archive_path, root)
        artifacts.append(
            {
                "archive_bytes": archive_path.stat().st_size,
                "archive_filename": archive_name,
                "archive_sha256": file_sha256(archive_path),
                "compression": config["compression"],
                "embedding_space_id": config["embedding_space_id"],
                "minimum_deployment_target": config["minimum_deployment_target"],
                "package_filename": package.name,
                "quantization": config["quantization"],
                "recommended": config["recommended"],
                "variant": variant,
                **details,
            }
        )

    manifest = {
        "artifacts": artifacts,
        "base_model": {
            "license": "Apache-2.0",
            "repository": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_filename": "model.safetensors",
            "weights_sha256": "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a",
        },
        "contract": {
            "external_preprocessing": {
                "id": "exif-rgb-objectcrop-square-edge-pad-bicubic384-v1",
                "steps": [
                    "apply EXIF orientation and convert to RGB",
                    "crop the target object",
                    "center in a square padded with the per-channel median of outer-edge pixels; put an odd extra pixel on the bottom or right",
                    "bicubic-resize to 384x384",
                ],
            },
            "input": {
                "bias": [-1.0, -1.0, -1.0],
                "color": "RGB",
                "name": "image",
                "scale": 1.0 / 127.5,
                "size": [384, 384],
                "type": "CVPixelBuffer",
            },
            "output": {
                "dtype": "float32",
                "l2_normalized": True,
                "name": "embedding",
                "shape": [1, 1152],
            },
        },
        "conversion": {
            "coremltools": "9.0",
            "format": "mlprogram",
            "python": ">=3.11,<3.12",
            "source_tag": f"v{args.release_version}",
            "torch": "2.7.0",
            "transformers": "4.48.1",
            "uv_lock_sha256": file_sha256(root / "uv.lock"),
        },
        "license": "Apache-2.0",
        "license_files_in_archives": ["LICENSE", "NOTICE"],
        "release_version": args.release_version,
        "repository": REPOSITORY,
        "results": "RESULTS.json",
        "schema_version": 1,
        "verification": {
            "compute_units": "CPU_ONLY",
            "reference_file": "tests/reference_outputs.npz",
            "reference_sha256": file_sha256(root / "tests/reference_outputs.npz"),
            "script": "verify.py",
        },
    }
    manifest_path = dist / "MODEL_MANIFEST.json"
    write_json(manifest_path, manifest)
    shutil.copy2(root / "RESULTS.json", dist / "RESULTS.json")
    checksums = "".join(
        f"{item['archive_sha256']}  {item['archive_filename']}\n" for item in artifacts
    )
    checksums += f"{file_sha256(manifest_path)}  MODEL_MANIFEST.json\n"
    checksums += f"{file_sha256(root / 'RESULTS.json')}  RESULTS.json\n"
    checksum_path = dist / "SHA256SUMS"
    checksum_path.write_text(checksums)
    if args.write_root_metadata:
        shutil.copy2(manifest_path, root / "MODEL_MANIFEST.json")
        shutil.copy2(checksum_path, root / "SHA256SUMS")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
