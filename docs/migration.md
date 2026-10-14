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
