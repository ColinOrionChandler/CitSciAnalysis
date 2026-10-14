# Workflows

- `testing-and-coverage.yml`: supported Python/platform tests, coverage floor,
  minimum direct dependencies, and optional activated Codecov uploads.
- `pre-commit-ci.yml`: Ruff, strict mypy, wheel installation, and source-distribution rebuilding.
- `build-documentation.yml`: warning-free Sphinx build and downloadable HTML artifact.
- `smoke-test.yml`: daily dependency compatibility tests and manual runs.
- `benchmarks.yml`: synthetic ASV measurements, PR comparison, retained history,
  and GitHub Pages publication on main.
- `publish-to-pypi.yml`: future version-tagged GitHub releases only; requires the
  PyPI trusted publisher described in `docs/releases.md`.

All ten test/quality/package/documentation checks are required to merge main.
Benchmark execution must succeed; timing comparisons are advisory.
See `docs/migration.md` for template provenance and pending account setup.
