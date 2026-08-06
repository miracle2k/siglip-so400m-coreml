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

`release.py` adds release metadata and creates deterministic ZIP archives:

```sh
uv run release.py \
  --p8 build/SigLIPSO400M384-P8.mlpackage \
  --p6 build/SigLIPSO400M384-P6.mlpackage \
  --dist dist
```

## Results

On a small private regression suite, P8 retained the PyTorch baseline's 83.12%
top-1 and 94.80% top-5 retrieval recall. P6 reached 82.00% and 94.49%.
Median warm inference on an M4 Pro was 119.6 ms for P8 and 114.1 ms for P6.
See [`RESULTS.json`](RESULTS.json) for the full aggregate report.

The suite contains only 86 crops and eight eligible identities. These results
are regression evidence, not a general accuracy claim, and are not physical
iPhone benchmarks. Both release packages also pass Core ML CPU execution and
Xcode 16.3 compilation for an iOS 17 deployment target.

## License

The pinned source checkpoint declares Apache-2.0. This repository contains a
modified conversion of that checkpoint and is also released under Apache-2.0.
See [`NOTICE`](NOTICE). This project is not affiliated with Google, Hugging Face,
or Apple.
