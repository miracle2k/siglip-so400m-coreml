# SigLIP SO400M 384 for Core ML

This repository records our Core ML conversion and Apple-platform validation
work for the pinned upstream checkpoint. Downloadable builds are published in
the [Hugging Face model repository](https://huggingface.co/metaclass/siglip-so400m-patch14-384-coreml).

## Downloads

| Build | Apple target | Status |
| --- | --- | --- |
| [P8 / r1](https://huggingface.co/metaclass/siglip-so400m-patch14-384-coreml/tree/main/p8/r1) | iOS 17+ | Recommended default |
| [P6 / r1](https://huggingface.co/metaclass/siglip-so400m-patch14-384-coreml/tree/main/p6/r1) | iOS 17+ | Smaller experimental alternative |
| [P6G16 / candidate.1](https://huggingface.co/metaclass/siglip-so400m-patch14-384-coreml/tree/main/p6g16/candidate.1) | iOS 18+ | Candidate; not yet validated on a physical iPhone |

Each artifact directory contains its archive, `manifest.json`, and
`SHA256SUMS`. Pin one exact artifact path and its archive SHA. The variants use
different embedding spaces, so their vectors must not be mixed.

No published build has yet been validated on a physical iPhone.

## Results

P8 is the current iOS 17 default. P6 is retained for smaller-package
experiments, while P6G16 remains an iOS 18 candidate. The concise
Apple-platform decision history is in
[EXPERIMENTS.md](EXPERIMENTS.md); portable aggregate measurements and published
artifact hashes are in [RESULTS.json](RESULTS.json).

## Scope and license

The source repository contains the conversion and validation tooling; model
archives live only on Hugging Face. The pinned upstream checkpoint and this
repository are released under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
