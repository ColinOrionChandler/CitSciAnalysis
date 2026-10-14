# LINCC template migration

## Provenance and local choices

Generated with Copier 9.19.0 from LINCC Python Project Template v2.3.0, using the
committed `.copier-answers.yml`. The integration baseline is upstream `cd96e14`,
including pandas 3 score finalization and candidate worksets. Existing Git history
and historic scripts are preserved.

The template's repository name is `CitSciAnalysis`; the distribution override is
`citsci-analysis`. Keep that distinction when updating. Existing exports, scorer
entry points, and direct consolidation execution replace template example code
and its generated version CLI. `__version__` is additive.

Other intentional adaptations: Python 3.11–3.14 plus macOS 3.12 CI; strict mypy
covers tests and benchmarks; no notebooks/Jupyter dependency; dependency metadata
is centralized in pyproject.toml; requirements.txt installs the local project;
historic scripts are excluded from formatting, typing, and distributions; docs
build with warnings as errors; benchmark failures propagate; coverage uses the
measured baseline; benchmark history starts at this migration; no PR-comment bot.

## Update the template

Start with a clean feature branch, install Copier >=9.1, and run `copier update`
with the desired pinned release. Inspect every changed file and resolve conflicts,
especially pyproject metadata, package exports, workflows, and README preservation.
Run the full development checks and installed-artifact tests before merging.
Never run the new-project initializer against this existing repository.

## Services

- Source: https://github.com/ColinOrionChandler/CitSciAnalysis
- Planned docs: https://citsci-analysis.readthedocs.io/
- Coverage: https://app.codecov.io/gh/ColinOrionChandler/CitSciAnalysis
- Benchmarks: https://colinorionchandler.github.io/CitSciAnalysis/
- Future package: https://pypi.org/project/citsci-analysis/

Service activation and validation are recorded below after deployment. Links alone
are not proof that a service is connected. Read the Docs import and PyPI trusted
publisher registration may require user account authorization.

## Attribution

This project uses the [LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template).
See its [citation guidance](https://lincc-ppt.readthedocs.io/en/latest/source/citation.html).

## Validation evidence (2026-10-14)

- Baseline `cd96e14`: 69 upstream tests, 86.2464% statement coverage.
- Migration: 73 tests including four installed-package checks. Maintained-code
  coverage is above baseline; the required floor is 86.24% (baseline truncated to
  two decimals), with Codecov configured to compare against the base revision.
- Synthetic 90-vote fixtures were run against an untouched archive of the baseline
  and the migrated code. Naive, time, and legacy modes were checked with both CSV
  and Parquet; Rubin used nine chunks including duplicate votes and independently
  missing metadata. All output CSV bytes matched, including duplicate COC sidecar
  handling. Diagnostics matched after removing only the temporary sidecar path.
- Strict mypy covers 14 source/test/benchmark files; Ruff lint and formatting pass.
- CI passes all five Python/platform combinations and Python 3.11 with the lowest
  direct dependencies. Wheel and sdist installations are tested outside the source
  tree; the sdist is rebuilt without Git. Metadata and license validation pass.
- Sphinx builds with warnings as errors. Navigation, installation examples, and
  generated API pages were visually inspected in Chrome.
- All six synthetic ASV benchmarks execute locally and in GitHub Actions. PR
  comparison is advisory; benchmark execution failures propagate.
- The public-visibility audit covered 101 historical blobs: no credential-pattern
  findings or research-data files. Existing local/remote path defaults remain for
  compatibility; no credentials are embedded. Historical scripts remain unchanged.

## Account activation status

Read the Docs activation is pending by the owner's explicit choice. The Sphinx
configuration and documentation-build CI are complete. Import the public repository
as `citsci-analysis` in Read the Docs to activate the planned documentation URL.

PyPI's GitHub `pypi` environment permits version tags only. Register the pending
trusted publisher described in [Releases](releases.md) before the first release.
No version tag, GitHub release, or PyPI package was published by this migration.

Codecov account activation is also pending: an authenticated OIDC upload returned
`Repository not found`. Required tests still enforce the 86.24% coverage floor and
save coverage.xml as a CI artifact. After activating this repository in Codecov,
set the GitHub Actions repository variable `CODECOV_ENABLED=true` and rerun Tests.
Enabled uploads fail visibly on errors. No static upload token is required because
the workflow uses GitHub OIDC. Codecov is not reported as operational yet.

GitHub repository visibility is public, Dependabot vulnerability alerts are enabled,
and main requires all ten test/quality/package/documentation checks through a PR,
with zero mandatory reviewer approvals. The migration merged through
[PR #1](https://github.com/ColinOrionChandler/CitSciAnalysis/pull/1) at `7e7ce50`;
all required checks passed before merge and all main workflows passed afterward.

The [public benchmark dashboard](https://colinorionchandler.github.io/CitSciAnalysis/)
is live. GitHub Pages deployment succeeded, and its rendered benchmark list was
inspected with all six runtime/peak-memory measurements for the migration baseline.
Results are retained on the `benchmarks` branch; generated HTML is served from
`gh-pages`. Timing comparisons remain advisory. Read the Docs, Codecov, and PyPI
account activation remain pending as described above.
