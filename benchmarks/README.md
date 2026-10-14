# Benchmarks

Run from this directory: `asv machine --yes`, `asv run HEAD^!`, and `asv publish`.
Use `asv run --quick --python=same HEAD^!` for a local smoke check after installing
the package. CI measures runtime and peak memory on Python 3.12. Data is synthetic.
Main builds save history on the benchmarks branch and publish to GitHub Pages.
PRs compare their base and current revision; timing changes are advisory. Execution
failures are errors. Results start at the migration baseline, not all historical commits.
