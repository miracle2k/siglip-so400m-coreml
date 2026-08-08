# Apple-platform conversion record

This is the concise decision history for converting the pinned upstream
checkpoint to Core ML. Published artifact hashes and aggregate measurements are
in [RESULTS.json](RESULTS.json); artifact manifests are on the model repository.

## Retained builds

| Build | Apple target | Outcome |
| --- | --- | --- |
| FP16 reference | iOS 17 | Valid fidelity baseline, not published because its 854.1 MB package was too large. |
| P8 / r1 | iOS 17 | Stable default. Scalar 8-bit palettes preserved the strongest fixed-regression result in the iOS 17 search. |
| P6 / r1 | iOS 17 | Smaller alternative. It passed conversion and packaging checks, but lost enough fixed-regression fidelity not to replace P8. |
| P6G16 / candidate.1 | iOS 18 | Grouped-channel 6-bit candidate. CPU execution, deterministic packaging, and iOS 18 compilation passed; physical-iPhone validation remains outstanding. |

P6G16 has its own embedding space and uses a separate iOS 18 FP16
intermediate, so it does not change the existing iOS 17 P8/P6 graph contracts.

## Decisions from the search

- Keep P8 as the default: it gave the best retained size/fidelity balance for
  iOS 17.
- Keep P6 as opt-in only: it reduced package size, but missed the default
  fidelity budget.
- Keep P6G16 as a candidate: it is smaller than P8 and cleared the retained
  non-device gates, but must not be promoted without physical-iPhone evidence.
- Reject smaller grouped P6 defaults: group size 4 added unacceptable Core ML
  latency, while group size 8 missed the fixed-regression gate.
- Reject vector palettes, grouped P4, blockwise INT4, and the tested linear
  INT8 paths: each damaged conversion fidelity beyond the retained budget.
- Reject scale-enabled direct Core ML packaging and Torch post-training
  palettes: the former inflated packages and the latter failed Core ML parity
  or performance checks.
- Reject the affected iOS 18 grouped-UINT4 GPU path: its output diverged from
  the decompressed model, so it is not a release path.

## Evidence limits

The fixed regression suite is retained only as aggregate conversion evidence;
it is not a general-accuracy or physical-device benchmark. P6G16's published
candidate is a pinned artifact: a fresh k-means run can choose a different,
near-equivalent palette. A fresh P6G16 build therefore requires validation and
a new artifact revision rather than replacement of `candidate.1`.
