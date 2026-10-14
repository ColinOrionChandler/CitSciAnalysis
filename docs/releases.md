# Releases

Versions come from Git tags through setuptools-scm. Untagged builds are development
versions; source distributions carry version metadata for rebuilding without Git.
An unbuilt source checkout reports `0.0.0+unknown` if no generated version exists.

## One-time setup

Register a pending PyPI trusted publisher for `citsci-analysis`, repository
`ColinOrionChandler/CitSciAnalysis`, workflow `publish-to-pypi.yml`, environment
`pypi`. This requires a PyPI account with access to the project name. No API token
belongs in this repository. A 404 from the PyPI JSON API is not a reservation.

## Publish a future release

1. Complete required PR checks, update the changelog, and merge the release changes.
2. Create a version tag such as `v0.1.0` on that verified main commit.
3. Publish the corresponding GitHub release. The workflow verifies tag syntax,
   main ancestry, package metadata, tests, and wheel/sdist contents before upload.
4. Confirm the version on PyPI and install its wheel in a clean environment.

Ordinary branch pushes, PRs, and merges never upload a package. This migration
creates no release tag and uploads no package. Do not overwrite a published version.
