# Benchmarks

Run from this directory: `asv machine --yes`, `asv run HEAD^!`, and `asv publish`.
Use `asv run --quick --python=same` for a local smoke check after installing
the package. CI measures runtime and peak memory on Python 3.12. Data is synthetic.
Main builds save history on the benchmarks branch and publish to GitHub Pages.
PRs compare their base and current revision; timing changes are advisory. Execution
failures are errors. Results start at the migration baseline, not all historical commits.

The Rubin workload contains 2,000 repeated user/review-unit pairs in a later chunk.
The audit corrected the original workload, which had no duplicate pairs. ASV hashes
benchmark and setup code, so the corrected scoring measurements start a new series;
the original baseline remains in the results branch and registry history continues.
See [ASV benchmark versioning](https://asv.readthedocs.io/en/stable/writing_benchmarks.html#benchmark-versioning).
PR artifacts and the Actions summary include the base/head comparison. Failed runs
retain available artifacts and report failure explicitly.
