# Development

Install with `python -m pip install -e '.[dev]'` and run `bash .setup_dev.sh`.

Required checks:

```sh
ruff check .
ruff format --check .
mypy
pytest --cov=citsci_analysis
python -m build
python -m twine check dist/*
python tools/validate_distributions.py dist
python tools/check_scientific_parity.py
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Strict typing covers source, tests, benchmarks, and validation tools. Historical scripts are archived
and excluded. Tests and benchmarks use synthetic inputs. Preserve scoring and
workset contracts; see the repository DEVELOPMENT_GUIDE.md for contribution rules.

See [migration notes](migration.md) for template updates and
[release procedures](releases.md) for future publishing.
