# Publishing model artifacts

This repository is source and evaluation only: never commit model archives,
manifests, or checksums.

For a new `<variant>/<revision>`, from a clean committed checkout:

    uv run release.py --variant <variant> --source <model.mlpackage> \
      --artifact-revision <revision> --status <status> \
      --output dist/<variant>/<revision>
    (cd dist/<variant>/<revision> && shasum -a 256 -c SHA256SUMS)
    huggingface-cli upload metaclass/siglip-so400m-patch14-384-coreml \
      dist/<variant>/<revision> <variant>/<revision> --repo-type model \
      --commit-message "Publish <variant>/<revision>"

First confirm the Hub path is absent; after upload, record its Hub commit and
check the Hub LFS archive SHA matches manifest.json. Never replace an artifact:
rebuilds receive a new revision. Apps pin the exact path and archive SHA.
