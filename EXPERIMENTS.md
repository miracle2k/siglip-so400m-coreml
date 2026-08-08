# Experiment record

This is the retained, public-safe decision trail for the Core ML conversion of
`google/siglip-so400m-patch14-384` at revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. Detailed measurements are in
[`RESULTS.json`](RESULTS.json); this document explains which experiments changed
the shipped design and which were deliberately not carried forward.

## Retained variants

| Variant | Deployment target | Source package | Decision |
|---|---|---:|---|
| FP16 | iOS 17 | 854.1 MB | Fidelity reference; not distributed because of size. |
| P8 / r1 | iOS 17 | 427.8 MB | Stable default. It preserved the private regression baseline closely and is the safest compatibility choice. |
| P6 / r1 | iOS 17 | 321.1 MB | Size-first opt-in alternative, not the default: it loses measurable private retrieval fidelity. |
| P6G16 / candidate.1 | iOS 18 | 324.4 MB | Candidate. It uses 6-bit scalar palettes independently over output-channel groups of 16. It passed the current regression gates and public ImageNet-v2 check, but still needs physical-iPhone validation. |

P6G16 has a distinct `embedding_space_id`; its vectors must never be mixed with
P8 or P6 vectors in one index. Its grouped-palette encoding requires an
independent iOS 18 FP16 intermediate, so adding it cannot change the released
iOS 17 P8/P6 graph contracts.

## Decisions from the compression search

- **Keep P8 as the deployment default.** Its 49.9% size reduction from FP16
  retained the strongest practical quality/size/latency balance in the initial
  iOS 17-compatible search.
- **Keep P6 available, but do not promote it by default.** It is smaller and a
  little faster than P8, but its private retrieval regression is outside the
  default-quality budget.
- **Keep P6G16 as a candidate.** It is 24.2% smaller than P8, near P8
  latency on the test M4 Pro, and scored 77.27% top-1 on the pinned public
  ImageNet-v2 protocol (P8: 77.02%; FP16: 77.16%). That is useful corroboration,
  not proof of a general improvement over FP16 or a substitute for device tests.
- **Do not pursue smaller grouped P6 variants as defaults.** Group size 4 had
  good fidelity but materially worse latency; group size 8 narrowly missed the
  retrieval gate. Vector palette clustering reduced size sharply but damaged
  embedding geometry and top-5 retrieval.
- **Reject 4-bit and linear quantization for this checkpoint.** Grouped P4 and
  blockwise INT4 did not retain enough embedding fidelity. The three tested
  linear INT8 configurations lost roughly 8–9 percentage points of private
  top-1 retrieval, unlike scalar k-means palettization.
- **Reject scale-enabled and Torch post-training palette paths.** Direct
  scale-enabled Core ML packages grew dramatically, while the Torch
  post-training path failed the Core ML parity guard and was much slower.
- **Treat the affected iOS 18 GPU path as unsafe.** On the test macOS/Core ML
  stack, grouped UINT4 output on GPU-backed compute units diverged from the
  decompressed model. CPU plus Neural Engine was correct, but the candidate
  still failed fidelity gates, so it is not a release path.

## Evidence and limits

The private regression suite used 86 frozen crops and 20 deterministic retrieval
splits. It is useful for preventing known regressions but is too small to make a
general accuracy or physical-device claim. Only aggregate values are published;
no private images, vectors, object identities, boxes, or evaluation tooling are
part of this repository.

The public complement is [`benchmark_imagenet_v2.py`](benchmark_imagenet_v2.py):
a pinned 10,000-image ImageNet-v2 zero-shot protocol using a pinned Big Vision
checkout and 81 CLIP-paper prompts. It is reproducible from public inputs and
reports package hashes, paired accuracy deltas, output cosine, and latency.

P6G16 packaging is reproducible from the pinned source package, rather than
byte-reproducible from a fresh k-means run: parallel k-means can select a
different but near-equivalent palette. `release.py` therefore validates the
candidate's source package tree, graph, and weights before staging an archive.

## What is intentionally absent

The original private fixture builder, retrieval evaluator, remote-machine runner,
raw candidate vectors, packages from rejected trials, and machine-specific
experiment scripts are intentionally omitted. They were valuable for the search,
but would not make this public repository reproducible or understandable. The
repository retains only the conversion/packaging mechanics for P8, P6, and P6G16,
the public benchmark, compact aggregate results, and the decisions above.
