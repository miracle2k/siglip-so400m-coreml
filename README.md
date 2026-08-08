# SigLIP SO400M 384 for Core ML

Unofficial Core ML conversion and reproducibility tooling for the vision tower
from [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
at revision 9fdffc58afc957d1a03a25b10dba0329ab15c2a3.

## Model artifacts

This is a source repository. It intentionally contains no model packages,
release ZIPs, global manifests, or checksums. Each downloadable Core ML build
is independently published in the project Hugging Face model repository at:

    <variant>/<artifact-revision>/

Each directory contains one archive, manifest.json, and SHA256SUMS. Once
published, it is not replaced: a rebuilt model gets a new artifact revision.

| Artifact | Runtime | Use |
|---|---|---|
| P8 / r1 | iOS 17+ | Recommended default |
| P6 / r1 | iOS 17+ | Smaller experimental alternative |
| P6G16 / candidate.1 | iOS 18+ | Experimental candidate; not yet physically validated on iPhone |

P8, P6, and P6G16 are independent variants of the same upstream model, not
successive model versions.

## Contract

- Input image: one 384 by 384 RGB CVPixelBuffer.
- In-model normalization: x / 127.5 - 1.
- Output embedding: Float32 [1, 1152], L2-normalized.
- This is an image encoder only; it does not include SigLIP's text tower.

Our retrieval pipeline applies EXIF orientation, converts to RGB, crops the
object, centers it in a square padded with the per-channel median colour of its
outer-edge pixels, and bicubic-resizes to 384 by 384. Index and query images
must use the same preprocessing and exact artifact. In particular, never mix
embedding spaces across P8, P6, or P6G16.

## iOS use

Choose one exact artifact, verify its archive using the SHA256SUMS beside it,
then add the extracted mlpackage to Xcode or compile it with
MLModel.compileModel(at:). Persist the artifact's embedding_space_id alongside
the vector index. Do not update to a different variant automatically.

## Reproduce

Conversion requires macOS. The model revision and Python toolchain are pinned.
A clean build downloads the 3.51 GB upstream checkpoint and needs several
additional gigabytes for intermediate packages.

    uv sync --frozen
    uv run convert.py --output build --variants p8 p6
    uv run verify.py build/SigLIPSO400M384-P8.mlpackage \
      build/SigLIPSO400M384-P6.mlpackage

P6G16 uses a separate iOS 18 FP16 intermediate:

    uv run convert.py --output build-p6g16 --variants p6g16

Package one immutable artifact:

    uv run release.py \
      --variant p8 \
      --source build/SigLIPSO400M384-P8.mlpackage \
      --artifact-revision r1 \
      --status stable \
      --output dist/p8/r1

The output directory contains only the archive, manifest.json, and SHA256SUMS.
Copy it unchanged to a new matching Hugging Face path.

The public zero-shot regression harness requires a supplied public ImageNet-v2
archive and a pinned Big Vision checkout. It never uses the private retrieval
fixture:

    uv run --frozen benchmark_imagenet_v2.py --help

See [EXPERIMENTS.md](EXPERIMENTS.md) and [RESULTS.json](RESULTS.json) for the
decision record and evaluation limits.

## Results

The private regression suite has 86 crops and eight eligible identities. It is
useful for catching known regressions, not for general accuracy claims or
physical-iPhone performance. P8 remains the default. P6G16 has passed CPU
execution, deterministic packaging, and iOS 18 compilation, but still requires
physical-iPhone validation.

The public ImageNet-v2 zero-shot protocol corroborates that P8 and P6G16 retain
the published checkpoint's general semantic quality. It does not measure
same-instance photo retrieval or device performance.

## License

The pinned source checkpoint declares Apache-2.0. This repository contains a
modified conversion of that checkpoint and is also released under Apache-2.0.
See [NOTICE](NOTICE). This project is not affiliated with Google, Hugging Face,
or Apple.
