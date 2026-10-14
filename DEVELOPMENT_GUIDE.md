# Development guide

Use Python 3.11–3.14 and an isolated environment. Run `bash .setup_dev.sh` to install
editable development dependencies, documentation tools, and pre-commit hooks.

## Scientific contracts

Preserve scoring equations, defaults, CLI arguments, output schemas, duplicate-vote
handling, and independent first-non-null Rubin metadata aggregation. The current
workset contracts include resumable, exclusive publication and source preservation.
Use synthetic data in tests and benchmarks. Never add local research inputs,
results, credentials, or downloaded images. Do not modify `historic_code/`.

## Checks

```sh
ruff check .
ruff format --check .
mypy
pytest --cov=citsci_analysis --cov-report=term-missing
python -m build
python -m twine check dist/*
python tools/validate_distributions.py dist
python tools/check_scientific_parity.py
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Strict mypy covers maintained source, tests, benchmarks, and validation tools. Third-party Arrow
imports have a scoped missing-stubs exception; do not disable checks project-wide.
Keep formatting/typing separate from behavior changes. Use a feature branch and
pull request; all required checks must pass before merging, without a mandatory
second reviewer. Performance changes are advisory, but failed benchmarks are errors.

## Template and releases

See [migration notes](docs/migration.md) for Copier customizations and service setup,
and [release procedures](docs/releases.md) before publishing. No package is
published by ordinary pushes or merges. Preserve distribution `citsci-analysis`,
module `citsci_analysis`, and repository `ColinOrionChandler/CitSciAnalysis`.
