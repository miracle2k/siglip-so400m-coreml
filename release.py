#!/usr/bin/env python3
"""Create one immutable Core ML artifact directory.

The source repository contains conversion and verification tooling. Every
deployable build is packaged independently as <variant>/<revision>/ with its
own archive, manifest, and checksums; it is not a repository-wide release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import coremltools as ct

MODEL_ID = "google/siglip-so400m-patch14-384"
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
SOURCE_REPOSITORY = "https://github.com/miracle2k/siglip-so400m-coreml"
ARTIFACT_STATUSES = ("stable", "experimental", "candidate")
VARIANTS = {
    "p8": {
        "bits": 8,
        "compression": "kmeans-8bit-scalar-per-tensor",
        "default_status": "stable",
        "minimum_deployment_target": "iOS17",
        "model_graph_sha256": "2f64d448265107972b2c41fa7a13f4e436b76c4f819e23d9131f1c7b26b32472",
        "quantization": "palette-kmeans-8bit-per-tensor",
        "weights_sha256": "0e84cef4bcb2fd9c4eee9496809248463998506c7185f57e720a44f874aa04d9",
        "filename": "SigLIPSO400M384-P8.mlpackage",
        "legacy_archive": {
            "archive_bytes": 403610178,
            "archive_sha256": "28fd93f254d56dcfa2c9155c7d65d86d9fe63d688a12012621660f17db9f0738",
            "package_bytes": 427777672,
            "package_tree_sha256": "4fd72827155aedb94ab0a412a2b9fe8f38a31eb1e250c325f9773d130697680f",
            "source_commit": "f8c6a35a2d1535e65d2efe58b68bea2807efb34b",
            "source_package_tree_sha256": "8ab594c40abb7b9f658ce38e812de82ca70a5407f54e9c2dd471567754f8aa40",
        },
        "recommended": True,
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p8-kmeans8pt-v1"
        ),
    },
    "p6": {
        "bits": 6,
        "compression": "kmeans-6bit-scalar-per-tensor",
        "default_status": "experimental",
        "minimum_deployment_target": "iOS17",
        "model_graph_sha256": "b888adaf476936ca40eb28b3f27c2b005163d341040d3f415588078a12e11dfd",
        "quantization": "palette-kmeans-6bit-per-tensor",
        "weights_sha256": "d583e30c3583d935ba5b15c3c0adfdbb172779aee40bf63f8973017c0cd194fd",
        "filename": "SigLIPSO400M384-P6.mlpackage",
        "legacy_archive": {
            "archive_bytes": 314336486,
            "archive_sha256": "ae94fa8f3d992e70a58fb05b642f408fe5caae317f1dbe988c8b056a220cc045",
            "package_bytes": 321091552,
            "package_tree_sha256": "1974e6bcbdda786c0c92e1133cc82973d50e7870375ad30e43f3bcf3122ca58e",
            "source_commit": "f8c6a35a2d1535e65d2efe58b68bea2807efb34b",
            "source_package_tree_sha256": "429cf60610f26120c485ae7be19c966731381a9ce0b5f2d6b5258b5ab3e53193",
        },
        "recommended": False,
        "embedding_space_id": (
            "siglip-so400m-p14-384@9fdffc58:pooler-l2:"
            "rgb-square-bicubic384-v1:coreml-p6-kmeans6pt-v1"
        ),
    },
    "p6g16": {
        "bits": 6,
        "compression": "kmeans-6bit-scalar-per-grouped-channel-16",
        "default_status": "candidate",
        "minimum_deployment_target": "iOS18",
        "model_graph_sha256": "c9de8d03751a01339314e70d76c6af8b276c21ee578570dbe075f581ca5019a8",
        "quantization": "palette-kmeans-6bit-grouped-channel-16",
        "required_source_package_tree_sha256": "708fd43a1ed2f9ca1d5097c97c5890fe3d5551c4a31c9df1bdedc55826c17b21",
        "weights_sha256": "782c2b3c305b8826151f334a93d0de558a65d705963f138d17ab9c5d540e0843",
        "filename": "SigLIPSO400M384-P6G16.mlpackage",
        "legacy_archive": {
            "archive_bytes": 319112765,
            "archive_sha256": "fdbc316249c5f25b05ccdc31a07ceb284e8827b38073ed20941352e2da2aba50",
            "package_bytes": 324372192,
            "package_tree_sha256": "26cd912a5062d67a30c6bea0db3f2cfe86575207ee59bfbf8238920232d3bb56",
            "source_commit": "24b13c3cb407df589f32d66645d9e8ce8e440810",
            "source_package_tree_sha256": "708fd43a1ed2f9ca1d5097c97c5890fe3d5551c4a31c9df1bdedc55826c17b21",
        },
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
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
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
    if metadata.userDefined.get("minimum_deployment_target") != config[
        "minimum_deployment_target"
    ]:
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
    artifact_id: str,
    artifact_revision: str,
    artifact_status: str,
    source_commit: str,
) -> dict[str, Any]:
    source, specification, source_tree_hash, source_weight_hash, graph_hash = validated
    destination = destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Source and destination package paths must not overlap")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact package: {destination}")

    config = VARIANTS[variant]
    metadata = specification.description.metadata
    metadata.author = "miracle2k"
    metadata.license = "Apache-2.0; derived from google/siglip-so400m-patch14-384"
    metadata.versionString = artifact_id
    for key in ("release_repository", "release_version"):
        metadata.userDefined.pop(key, None)
    metadata.userDefined.update(
        {
            "artifact_id": artifact_id,
            "artifact_revision": artifact_revision,
            "artifact_status": artifact_status,
            "conversion_source_commit": source_commit,
            "conversion_source_repository": SOURCE_REPOSITORY,
            "embedding_space_id": str(config["embedding_space_id"]),
        }
    )
    shutil.copytree(source, destination)
    model_path(destination).write_bytes(specification.SerializeToString(deterministic=True))
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
        "bytes": tree_size(destination),
        "filename": destination.name,
        "model_graph_sha256": graph_hash,
        "source_tree_sha256": source_tree_hash,
        "tree_sha256": tree_sha256(destination),
        "weights_sha256": release_weight_hash,
    }


def deterministic_zip(package: Path, output: Path, root: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary archive path already exists: {temporary}")
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
            with source.open("rb") as reader, archive.open(
                info, "w", force_zip64=True
            ) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    temporary.replace(output)


def validate_archive(path: Path, package_filename: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"Corrupt archive member in {path.name}: {broken}")
        names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"Archive has duplicate members: {path.name}")
    expected = {
        "LICENSE",
        "NOTICE",
        f"{package_filename}/Manifest.json",
        f"{package_filename}/Data/com.apple.CoreML/model.mlmodel",
        f"{package_filename}/Data/com.apple.CoreML/weights/weight.bin",
    }
    if set(names) != expected:
        raise ValueError(
            f"Unexpected archive layout in {path.name}: "
            f"missing={sorted(expected - set(names))}, "
            f"extra={sorted(set(names) - expected)}"
        )


def legacy_package_details(archive: Path, variant: str) -> dict[str, Any]:
    config = VARIANTS[variant]
    legacy = config["legacy_archive"]
    if archive.stat().st_size != legacy["archive_bytes"]:
        raise ValueError(f"Unexpected legacy archive size: {archive.name}")
    if file_sha256(archive) != legacy["archive_sha256"]:
        raise ValueError(f"Unexpected legacy archive hash: {archive.name}")
    validate_archive(archive, str(config["filename"]))
    return {
        "bytes": legacy["package_bytes"],
        "filename": config["filename"],
        "model_graph_sha256": config["model_graph_sha256"],
        "source_tree_sha256": legacy["source_package_tree_sha256"],
        "tree_sha256": legacy["package_tree_sha256"],
        "weights_sha256": config["weights_sha256"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_commit(root: Path, supplied: str | None) -> str:
    if supplied is None:
        supplied = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", supplied):
        raise ValueError("Source commit must be a full 40-character lowercase SHA-1")
    subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{supplied}^{{commit}}"],
        check=True,
    )
    return supplied


def clean_source_commit(root: Path, supplied: str | None) -> str:
    head = source_commit(root, None)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("Refusing to package a source model from a dirty worktree")
    if supplied is not None and supplied != head:
        raise ValueError("--source-commit must match the clean HEAD when using --source")
    return head


def artifact_manifest(
    variant: str,
    revision: str,
    status: str,
    source_commit_sha: str,
    archive: Path,
    package: dict[str, Any],
) -> dict[str, Any]:
    config = VARIANTS[variant]
    artifact_id = f"{variant}-{revision}"
    return {
        "artifact": {
            "id": artifact_id,
            "recommended": config["recommended"],
            "revision": revision,
            "status": status,
            "variant": variant,
        },
        "archive": {
            "bytes": archive.stat().st_size,
            "filename": archive.name,
            "sha256": file_sha256(archive),
        },
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
        "license": "Apache-2.0",
        "license_files_in_archive": ["LICENSE", "NOTICE"],
        "package": {
            **package,
            "compression": config["compression"],
            "embedding_space_id": config["embedding_space_id"],
            "minimum_deployment_target": config["minimum_deployment_target"],
            "quantization": config["quantization"],
        },
        "provenance": {
            "conversion_source_commit": source_commit_sha,
            "conversion_source_repository": SOURCE_REPOSITORY,
            "coremltools": "9.0",
            "format": "mlprogram",
            "python": ">=3.11,<3.12",
            "torch": "2.7.0",
            "transformers": "4.48.1",
        },
        "schema_version": 1,
    }


def parse_revision(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Artifact revisions use lowercase letters, digits, dots, underscores, and hyphens"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=Path,
        help="Fresh .mlpackage to validate, stamp, and archive.",
    )
    source_group.add_argument(
        "--archive",
        type=Path,
        help="One of the three pinned legacy archives to preserve byte-for-byte.",
    )
    parser.add_argument("--artifact-revision", type=parse_revision, required=True)
    parser.add_argument("--status", choices=ARTIFACT_STATUSES)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New immutable artifact directory, for example dist/p8/r1.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source_commit_sha = (
        clean_source_commit(root, args.source_commit)
        if args.source is not None
        else source_commit(root, args.source_commit)
    )
    if args.archive is not None:
        expected_legacy_commit = str(VARIANTS[args.variant]["legacy_archive"]["source_commit"])
        if args.source_commit != expected_legacy_commit:
            parser.error(
                "--archive requires the pinned --source-commit "
                f"{expected_legacy_commit}"
            )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Temporary artifact directory already exists: {staging}")
    staging.mkdir()

    try:
        variant = args.variant
        config = VARIANTS[variant]
        revision = args.artifact_revision
        status = args.status or str(config["default_status"])
        archive_name = (
            f"{Path(str(config['filename'])).stem}-"
            f"{config['minimum_deployment_target']}-{revision}.mlpackage.zip"
        )
        archive_path = staging / archive_name

        if args.source is not None:
            package_path = staging / str(config["filename"])
            package = stamp(
                validate_source(args.source.expanduser(), variant),
                package_path,
                variant,
                f"{variant}-{revision}",
                revision,
                status,
                source_commit_sha,
            )
            deterministic_zip(package_path, archive_path, root)
        else:
            source_archive = args.archive.expanduser().resolve()
            package = legacy_package_details(source_archive, variant)
            shutil.copy2(source_archive, archive_path)

        validate_archive(archive_path, str(config["filename"]))
        manifest = artifact_manifest(
            variant, revision, status, source_commit_sha, archive_path, package
        )
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        (staging / "SHA256SUMS").write_text(
            f"{manifest['archive']['sha256']}  {archive_path.name}\n"
            f"{file_sha256(manifest_path)}  manifest.json\n"
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
