Include in this repo: concise summary of the history of key experiments attempted, both successes and failures, focused exclusively on "make this model run on iOS/Apple platforms".

Do not include one-off artifacts, temporary spec files or notes. No file paths or data referencing a particular developer environment. Do not talk about the models general API or purpose - focus on our contributions. Anything in here is either about how to recreate a particular model build, or data about how one performed.

Notes on how to contribute to the repo should be concise and only in AGENTS.md. README.md is for people wanting to consume the models and results, and explain where to find both.


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
