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

## Initial validation evidence (2026-10-14)

- Baseline `cd96e14`: 69 upstream tests, 86.2464% statement coverage.
- Migration: 73 tests including four installed-package checks. Maintained-code
  coverage is above baseline; the required floor is 86.24% (baseline truncated to
  two decimals), with Codecov configured to compare against the base revision.
- Synthetic 90-vote fixtures were run against an untouched archive of the baseline
  and the migrated code. Naive, time, and legacy modes were checked with both CSV
  and Parquet; Rubin used nine chunks with independently missing metadata.
  Actual duplicate-vote coverage was added in the final audit below. All output CSV bytes matched, including duplicate COC sidecar
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

## Final audit (2026-10-14)

The merged migration at `2c3189a` was reviewed against the full plan in
[audit PR #9](https://github.com/ColinOrionChandler/CitSciAnalysis/pull/9). The audit
found validation/reporting gaps and corrected them without changing scientific
implementation or historical code:

- The release version check inspected an editable install instead of the archives.
  `tools/validate_distributions.py` now checks the actual wheel and sdist, including
  matching versions, release tag, BSD license/attribution, package entry files,
  and `py.typed`. It rejects unsafe paths, research-data payloads, and excluded
  result/history directories. Negative tests exercise these guards. Future release
  runs also install both artifacts outside the checkout and rebuild the sdist.
- The original 90-row parity fixture did not repeat user/thumbnail pairs. The
  committed `tools/check_scientific_parity.py` uses 182 rows with 91 repeated pairs,
  runs against archived pre-migration commit `cd96e14`, and compares 75 output
  files. All modes, CSV/Parquet, deduplication on/off, COC duplicates, independent
  missing metadata, and chunk sizes 1, 11, and 64 are covered. It is required in
  the Packaging job and uses the same installed dependencies for both revisions.
- The Rubin benchmark now has 2,000 actual duplicate pairs across chunk boundaries.
  ASV versions its changed setup separately; original records are retained, while
  corrected scoring measurements start a new series. PR reports are rendered
  after both revisions run, the comparison appears in the Actions summary, and
  failed runs retain available artifacts with an explicit failure summary.
- Source-tree execution is now checked automatically from a clean Git archive,
  including fallback version metadata, the module CLI, and direct consolidation.
  Published metadata also declares the supported Python versions and typed status.
- The PR checklist now references the actual developer guide and scientific
  contracts; bug reports request synthetic examples. Strict mypy includes the
  new validation tools (18 checked Python files).

| Plan area | Audit evidence/status |
| --- | --- |
| Template and history | Answers match the separately generated v2.3.0 template; original Git history and `historic_code/` are preserved. |
| Scientific/API compatibility | Original tests and installed API/CLI checks pass; the expanded before/after comparison passes for all 75 files. |
| Packaging and release preparation | Wheel and sdist pass archive checks and Twine; installed-package and unbuilt-source checks pass. Release upload remains restricted to published, valid version-tagged releases on main history. |
| Quality and coverage | All 81 tests, Ruff, and strict mypy pass; measured coverage remains 86.38%, above the unchanged 86.24% baseline floor. |
| Documentation | Sphinx builds with warnings as errors; existing scientific explanations and historical benchmark results remain included. |
| CI and public repository | Python 3.11–3.14/Linux, 3.12/macOS, lowest direct dependencies, daily smoke schedule, and ten required PR checks are retained. No extra approving reviewer is required. |
| Benchmark hosting | Public Pages dashboard and retained results were verified; corrected PR reports include both revisions and main publication retains earlier records. |
| Deferred accounts | Read the Docs import, Codecov activation, and PyPI publisher registration remain pending by owner instruction; none is represented as operational. |
| Release restraint | No version tag, GitHub release, or package upload was created. |

To repeat the scientific audit, run `python tools/check_scientific_parity.py` from a
Git checkout with full history and installed runtime dependencies. For distribution
validation, build into a clean directory and run
`python tools/validate_distributions.py dist`; pass `--release-tag vX.Y.Z` when
checking a future release. These commands never upload artifacts.
