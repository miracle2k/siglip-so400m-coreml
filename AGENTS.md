# Model artifacts

This repository contains conversion source, tests, and evaluation notes only.
Never commit model packages, ZIPs, manifests, or checksums; use ignored dist/.

Publish each build to the Hugging Face model repository under a variant and
artifact-revision path with its archive, manifest.json, and SHA256SUMS. Build
from a clean committed source SHA. Never replace a published artifact; rebuilds
get a new revision. The manifest records source commit, upstream revision,
variant/status, iOS target, embedding_space_id, and hashes. Apps pin an exact
artifact path and archive SHA; never follow latest. Before upload, confirm the
target path is absent and record the resulting Hugging Face commit SHA.
