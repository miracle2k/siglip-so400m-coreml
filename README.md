# SigLIP SO400M 384 for Core ML

Unofficial Core ML conversion of the vision tower from
[`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384)
at revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.

The [v0.1.0 release](https://github.com/miracle2k/siglip-so400m-coreml/releases/tag/v0.1.0)
contains two distinct, tested models:

| Model | Size | Use |
|---|---:|---|
| P8 | 427.8 MB | Recommended; 8-bit scalar per-tensor k-means palette |
| P6 | 321.1 MB | Smaller experimental alternative; 6-bit palette |

FP16 and the unsuccessful linear-int8 experiments are intentionally not
distributed. Both released packages are ML Programs targeting iOS 17+.

## Contract

- Input `image`: one 384×384 RGB `CVPixelBuffer`.
- In-model normalization: `x / 127.5 - 1`.
- Output `embedding`: Float32 `[1, 1152]`, L2-normalized.
- This is an image encoder only; it does not include SigLIP's text tower.

Our retrieval pipeline applies EXIF orientation, converts to RGB, crops the
object, centers it in a square padded with the per-channel median colour of its
outer-edge pixels, and bicubic-resizes to 384×384. If padding is odd, the extra
pixel goes on the bottom or right. Index and query images must use the same
preprocessing and model variant. P8 and P6 vectors are different embedding
spaces and must never be mixed.

## iOS use

Download one release archive and verify it against `SHA256SUMS`. Unzip it and
either add the `.mlpackage` to Xcode, or compile a downloaded package with
`MLModel.compileModel(at:)`. Use the identifiers in `MODEL_MANIFEST.json` to
version the on-device model and vector index.

## Reproduce and verify

Conversion requires macOS. The model revision and complete Python toolchain are
pinned. A clean build downloads the 3.51 GB upstream checkpoint and needs
several additional gigabytes for intermediate packages.

```sh
uv sync --frozen
uv run convert.py --output build --variants p8 p6
uv run verify.py build/SigLIPSO400M384-P8.mlpackage \
  build/SigLIPSO400M384-P6.mlpackage
```

To build P6G16 from source, request it on its own (or alongside P8/P6). It
uses an independent iOS 18 FP16 intermediate, so it cannot change the existing
iOS 17 P8/P6 graph contracts:

```sh
uv run convert.py --output build-p6g16 --variants p6g16
```

`release.py` adds release metadata and creates deterministic ZIP archives:

```sh
uv run release.py \
  --p8 build/SigLIPSO400M384-P8.mlpackage \
  --p6 build/SigLIPSO400M384-P6.mlpackage \
  --release-version 0.1.0 \
  --dist dist
```

Release packaging writes its manifest and checksums into `dist` by default. A
maintainer must explicitly pass `--write-root-metadata`, together with P8, P6,
and P6G16, to replace the repository's published manifest and checksums.

P6G16 is built from a separate iOS 18 FP16 base and has a distinct embedding
space. Its staged v0.2.0 candidate package is deliberately not in the public
manifest or GitHub releases yet:

```sh
uv run release.py \
  --p6g16 /path/to/SigLIPSO400M384-P6G16.mlpackage \
  --release-version 0.2.0 \
  --dist dist/p6g16-v0.2.0-candidate
```

`benchmark_imagenet_v2.py` is a separate public zero-shot regression harness.
It requires a supplied public ImageNet-v2 archive and a pinned Big Vision
checkout; it never uses the private retrieval fixture. Run
`uv run --frozen benchmark_imagenet_v2.py --help` for its inputs. See
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the retained compression decision record
and the limits of the private and public measurements.

## Results

On a small private regression suite, P8 retained the PyTorch baseline's 83.12%
top-1 and 94.80% top-5 retrieval recall. P6 reached 82.00% and 94.49%.
Median warm inference on an M4 Pro was 119.6 ms for P8 and 114.1 ms for P6.
See [`RESULTS.json`](RESULTS.json) for the full aggregate report.

The suite contains only 86 crops and eight eligible identities. These results
are regression evidence, not a general accuracy claim, and are not physical
iPhone benchmarks. Both release packages also pass Core ML CPU execution and
Xcode 16.3 compilation for an iOS 17 deployment target.

### Public zero-shot regression

The public harness follows the pinned Big Vision ImageNet-v2 zero-shot
protocol: 10,000 matched-frequency images, 81 CLIP-paper prompt templates, and
the original square-resize/crop preprocessing. The [SigLIP paper](https://arxiv.org/pdf/2303.15343)
reports 77.20% top-1 for this checkpoint; both FP16 Core ML and the pinned
PyTorch reference reached 77.16% here, validating the protocol.

| Model | Source-package size | Top-1 | Median warm inference |
|---|---:|---:|---:|
| FP16 | 854.1 MB | 77.16% | 141.2 ms |
| Released P8 | 427.8 MB | 77.02% | 119.7 ms |
| P6G16 candidate (P6 grouped-16) | 324.4 MB | 77.27% | 125.7 ms |

P8 differed from FP16 by -0.14 percentage points (paired bootstrap 95% interval
-0.30 to +0.01). The iOS 18-only P6G16 source package differed from
FP16 by +0.11 points (-0.09 to +0.31), so this test does not establish an
improvement over FP16; it was +0.25 points against P8 (+0.02 to +0.47). It is
locally staged and verified as a v0.2.0 candidate, but is not released or added
to the manifest. Physical-iPhone checks remain before publication. P8 remains
the deployment default.

## License

The pinned source checkpoint declares Apache-2.0. This repository contains a
modified conversion of that checkpoint and is also released under Apache-2.0.
See [`NOTICE`](NOTICE). This project is not affiliated with Google, Hugging Face,
or Apple.
