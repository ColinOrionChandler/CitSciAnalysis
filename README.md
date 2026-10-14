# CitSciAnalysis

Tools for scoring Active Asteroids citizen-science classifications for likely
activity. Originally developed by Will Burris, Colin Chandler, and Chad Trujillo
(Chandler+ 2024, Chandler 2022).

The historic scripts are preserved in `historic_code/`. The modern scorer lives
in `src/citsci_analysis/` and accepts either the old reparsed CSV exports or the
new cleaned Parquet files produced by `Zooniverse Toolz`.

## Input Mapping

Old reparsed CSV fields map to the cleaned Parquet fields as follows:

| Old CSV | Cleaned Parquet | Notes |
| --- | --- | --- |
| `username` | `user_name` | Used as the primary volunteer key. |
| `value` | `answer` | Normalized to `1.0` for yes and `0.0` for no. |
| `language` | `user_language` | Not used by the scorer. |
| `thumbname` | `thumbname` | Normalized to basename without `.png`, `.fits`, or `.fits.fz`. |
| `train` | `train` | Parquet uses truthy strings for training images and null for non-training images. |
| `started_at`, `finished_at`, `retired`, `finished_workflow`, `selection_state`, `batch`, `subject_id`, `utc_offset`, `classification_id` | same or richer equivalent | The Parquet also includes `user_id`, workflow metadata, `filename`, `name`, `tnid`, and `class`. |
| `Unnamed: 0` | omitted | Arbitrary historical index column. |

## Scoring Modes

All modes use the same three metric components:

1. Training-image yes rate.
2. Experience, scaled as `log10(number_of_classifications) / 4`.
3. Caution against over-optimism, `1 - test_image_yes_rate`.

The default metric weights are the historic `7, 2, 1`.

- `naive`: computes one final volunteer weight from that volunteer's capped
  classification history, then scores thumbnails with those static weights.
- `time`: computes a cumulative volunteer weight at each classification row,
  preserving the time-domain idea without polynomial fitting.
- `legacy`: approximates the old fifth-degree polynomial user-weight behavior
  and defaults to the old image-scoring constraints: at least four total
  classifications per volunteer, first 15 votes per thumbnail, and exact 0/1
  shortcuts when all counted votes agree.

## Usage

Use Python 3.11–3.14. Install the package in an isolated environment:

```bash
python3 -m pip install .
```

Score the cleaned Parquet with the faster static-weight mode:

```bash
PYTHONPATH=src python3 -m citsci_analysis score \
  "/path/to/reparsed_z/cleaned_classifications.parquet" \
  --output-dir results \
  --label parquet_naive_2025_1_13 \
  --mode naive
```

This writes:

- `output_total_<label>.csv`: every normalized `thumbname`, `score`,
  `coc_score`, and thumbnail-summary columns.
- `output_<label>.csv`: only rows at or above the cutoff, default `0.473`, with
  the same score, `coc_score`, and thumbnail-summary columns.
- `user_weights_<label>.csv`: volunteer weight diagnostics.

By default, the classic scorer adds a `coc_score` column from the local COC
sidecar file at
`/Users/colinchandler/Dropbox/School/Research/Thumbnail Examination UI/consolidated_scores.csv`
when that file is present. Use `--coc-scores /path/to/consolidated_scores.csv`
to require a specific file, or `--no-coc-scores` to leave `coc_score` blank.

The classic `output` and `output_total` files recreate the Zooniverse Toolz
`thumbnail_summary.csv` fields locally: number of classifications, yes/no
counts, fraction yes, training flag, and class. These summary columns are based
on the raw normalized rows before score-time vote filters or duplicate-user
deduping are applied.

Score Rubin Comet Catchers data with chunked, object-level aggregation:

```bash
PYTHONPATH=src python3 -m citsci_analysis score \
  "/path/to/rubin_comet_catchers/reparsed_z/cleaned_classifications.parquet" \
  --output-dir results \
  --label rubin_2026_5_1 \
  --mode rubin \
  --chunk-size 100000
```

Rubin mode treats each current image as a `review_unit_id` and each `class` as
the `object_id`. If `class` is missing, it falls back to `subject_id`, then to a
deterministic filename prefix. It writes:

- `rubin_review_unit_scores_<label>.csv`: image/pair-level scores.
- `rubin_object_scores_<label>.csv`: object-level scores ranked by top-3 mean
  review-unit score, with mean, peak, vote count, and best review unit.
- `rubin_user_weights_<label>.csv`: volunteer weight diagnostics.

In an interactive terminal, `score` writes `tqdm` progress to stderr while
preserving the final JSON summary on stdout. Parquet inputs include total rows
and ETA; CSV inputs report rows and rate. Use `--progress always` to force
progress in logs, or `--progress never` to silence it.

Future paired Rubin subjects should include explicit subject metadata for
`object_id`, `review_unit_id`, `image_ids`, `image_count`, and ordered image
roles when order matters. Then the review unit can become the submitted pair
without changing object-level aggregation.

Compare a modern output file to a historic total output:

```bash
PYTHONPATH=src python3 -m citsci_analysis compare \
  "/path/to/output_total_2025_1_13.csv.gz" \
  results/output_total_parquet_naive_2025_1_13.csv \
  --diff-output results/diff_parquet_naive_vs_historic.csv
```

Inspect old CSV versus new Parquet columns and normalized counts:

```bash
PYTHONPATH=src python3 -m citsci_analysis inspect-inputs \
  "/path/to/reparsed.csv.gz" \
  "/path/to/reparsed_z/cleaned_classifications.parquet"
```

Consolidate recursive score CSVs from current headerless exports and older
`ThumbName`/`Confidence` tables:

```bash
PYTHONPATH=src python3 -m citsci_analysis consolidate-scores \
  "/path/to/score_csv_directory"
```

The same utility can be run directly from a working folder:

```bash
python /path/to/CitSciAnalysis/src/citsci_analysis/consolidate.py --progress always
```

This writes `consolidated_scores.csv` in the input directory by default, with
one row per thumbnail basename sorted by name. Use `--output` to choose a
different CSV path. The output columns are:

- `datetime`: parsed datetime when valid, blank for missing or malformed dates.
- `path`: original image path when the source file provided one.
- `thumbnail_basename`: image basename with the final thumbnail extension
  removed.
- `score`: numeric score or legacy `Confidence` value.

When duplicate basenames appear, dated rows beat undated or malformed-date rows,
and the newest valid datetime wins. If all duplicate rows are undated, the
deterministic last row by sorted input path and row number is kept. The output
file is skipped during recursive discovery so rerunning the command does not
ingest its previous result.

## Development Checks

Install developer tools with `python3 -m pip install -e ".[dev]"` and follow
[DEVELOPMENT_GUIDE.md](https://github.com/ColinOrionChandler/CitSciAnalysis/blob/main/DEVELOPMENT_GUIDE.md). Run the test suite:

```bash
PYTHONPATH=src python3 -m pytest
```

For a quick coverage audit:

```bash
PYTHONPATH=src python3 -m pytest --cov=citsci_analysis --cov-report=term-missing
```

The highest-value coverage areas are the scoring invariants: input
normalization, duplicate-vote handling, user-weight modes, Rubin review-unit to
object aggregation, CLI output routing, and comparison/report utilities.

## 2025-01-10 Batch-30 Benchmark

Using the Dropbox benchmark folder named `20250110 through batch 30`:

- Old `reparsed.csv.gz`: 11,075,378 rows, 765,176 normalized thumbnails,
  16,325 users, 511,688 training rows.
- New `cleaned_classifications.parquet`: 11,075,395 rows, 765,175 normalized
  thumbnails, 16,325 users, 511,687 training rows.
- The only normalized thumbnail present in the old CSV but absent from the new
  Parquet is `particle_34_883x1139x1503x1759`, which appears to be a stray
  non-Active-Asteroids answer.

Score comparison to historic `output_total_2025_1_13.csv.gz`, after normalizing
`thumb_name`/`thumbname`:

| Mode | Pearson r | MAE | RMSE | Rows >= 0.473 |
| --- | ---: | ---: | ---: | ---: |
| `naive` | 0.9979518125 | 0.0028204277 | 0.0076191460 | 10,482 |
| `time` | 0.9984509761 | 0.0020505621 | 0.0064994003 | 10,479 |
| `legacy` | 0.9995119018 | 0.0014952353 | 0.0037763988 | 10,502 |

The historic total output has 765,352 rows but only 765,175 unique normalized
thumbnail names, so the comparison utility averages duplicate historic rows by
thumbnail before computing metrics.

## Candidate review worksets

`score` now runs an optional candidate-workset stage after writing the original
score files. Disable it with `--no-workset`. The scoring functions and existing
CSV schemas are unchanged. Worksets go to `--workset-root`, then
`CITSCI_WORKSET_ROOT`, then an existing mounted Dropbox `all_interesting`
directory; without those destinations they go to `candidate_worksets` beside
local outputs. Dataset paths and CSVs are runtime inputs, not repository assets.

Prepare or preview existing scores without repeating scoring:

```sh
citsci-score prepare-workset /path/to/output_total.csv --project aa \
  --workset-root /path/to/all_interesting --label aa_export \
  --training-metadata /path/to/cleaned_classifications.parquet --preview

citsci-score prepare-workset /path/to/rubin_review_unit_scores.csv --project rcc \
  --workset-root /path/to/all_interesting --label rcc_export \
  --training-metadata /path/to/cleaned_classifications.parquet \
  --source-root /path/to/existing/rcc
```

Remove `--preview` to execute. `--source-root` and `--source-index` are repeatable;
`--data-root` maps relative AA archive index paths to mounted storage. CSV and
Parquet indexes provide `basename` and `relative_path` columns; paths are relative
to that data root. AA defaults
use the existing candidate folders and mounted SAFARI2 manifests when available.
The Arnor index is queried for selected objects before selecting neighbors.
Its default path mapping follows the verified SAFARI2 relocation from `/epyc`
to `/astro/store/epyc3`; explicit mappings override that default.
Use `--remote-path-map /old/archive=/current/archive` when a verified archive
relocation has made index prefixes stale; the mapping is recorded in the workset.
`--remote` and `--remote-index` override that source; `--no-remote` explicitly
selects a local-only inventory. RCC never downloads or copies images.

AA selects non-training images at `--cutoff` (default 0.473), and includes
`--context` cutouts on either side (default 3) from the archive, merging overlaps.
RCC selects by the top-three mean of non-training review-unit scores, or the mean
of all available units for objects with fewer than three. Its default 0.473 is
inherited from the scorer and is **not a calibrated RCC threshold**. Missing or
conflicting training metadata is reported and excluded. A dedicated
`training_subject` flag identifies negative as well as positive training controls.
For an export known to use the Zooniverse nullable convention, explicitly supply
`--training-null-is-nontraining`; this treats blank `train` values as non-training
when no dedicated flag exists. Unknown nonempty values still remain unresolved.
The selected convention is recorded in the workset fingerprint.

Each `<project>/worksets/<label>-<fingerprint>/` contains `objects.csv`,
`images.csv`, `reveal_scores.csv`, `issues.json`, `inventory.json`, and
`workset.json` (contract v1). Source score/training paths and a checksum of the
selected source inventory are recorded for reproducibility.
AA publishes ordinary `.png` images only, never FITS or `.png.gz` review products.
Compressed PNG sources are decompressed and validated before publication; small
CSV/JSON sidecars remain necessary for scores and queue loading.
AA PNGs live once at `aa/objects/<object_id>/<logical_filename>.png`.
AA selection groups processing variants by object ID and observation datetime
before counting the ±3 neighbors. An unambiguous DECam source exposure datetime
is preferred because historical filename times can describe the start or midpoint
of the same exposure. Otherwise selection uses the filename's existing 10-second
precision; it does not round to a minute or merge distinct known exposures.
Each group prefers `pI`, then `p2`, then other products. Ties prefer a qualifying
target, then the lexical filename. Target membership survives a variant change,
but image scores remain attached to their original filenames: an unscored `pI`
has blank image scores even when selected by a qualifying `p2`. Object scores
remain unchanged. The saved inventory records omitted variants and their sources.
Image IDs retain the existing filename-based contract for classification compatibility.
The deduplication policy participates in AA fingerprints, so regeneration creates
a new workset and preserves existing frozen manifests and source PNGs.
The Python helper `narrow_aa_workset(path, context=3)` derives a smaller workset
from a complete saved AA selection without rescoring, querying archives, copying
images, or modifying its parent. Existing derived worksets retain their progress.
Legacy worksets must be regenerated before narrowing because duplicate variants
may have consumed their context slots; deleting duplicate rows alone cannot restore
the missing distinct neighbors.
Explicit `--context 10` remains supported; RCC defaults and fingerprints are unchanged.
Reveal CSVs distinguish image `cs_score`, native-scale `coc_score`, and
`object_cs_score`. Unclassified context has blank image scores. RCC uses supplied
source roots and records unresolved images without manufacturing local paths.

Fingerprints include score/training content and selection/source settings.
A successfully inventoried AA selection and its checked source inventory are frozen
on retry, avoiding another full archive scan. RCC rechecks its configured folders
for newly arrived images while retaining exact membership. Each retry validates
existing images, copies only missing content, and updates receipts atomically. Verified
transfer staging copies are removed after publication, leaving one shared PNG;
conflicts and incomplete transfers remain preserved for diagnosis or retry.
Use a different label to request a fresh selection after archive changes.
`--transfer-batch-size` limits each AA batch to 512 paths by default, so a large
selection starts transferring before all archive paths have been checked. Each
batch has a space preflight. `--transfer-timeout` bounds total preflight and
transfer time across batches (600 seconds by default). Timed-out command groups are terminated
with their SSH children. `--inventory-timeout` separately bounds the index query.
A failed archive query cannot select substitute local neighbors:
only target images can be prepared until inventory succeeds. Check `summary.status`
and `issues.json`; a partial workset is not a completion certificate.

`--migrate-legacy` inventories top-level legacy PNGs in `migration/legacy.json`,
then moves verified images into object folders. Empty, invalid, conflicting, and
preexisting duplicate files remain preserved and reported. When an archive copy
recovers an empty or invalid legacy file, the journal records
`recovered_original_preserved` with the shared PNG checksum and original path.
No images are deleted
or silently overwritten. A per-project destination lock prevents concurrent writers; after
an ungraceful termination, verify that its writer is stopped before removing the
`.workset-<project>.lock` directory.

Open a delivered workset in the updated astro-stampede checkout:

```sh
astro-stampede --workset /path/to/workset.json --validate
astro-stampede --workset /path/to/workset.json
```

Validation accepts metadata-only worksets and reports unmatched image counts.
Review sessions require usable matched images. Scores remain hidden until Reveal.
Existing label stores must be selected explicitly through astro-stampede's storage
options; workset preparation never modifies classification state.

## Project infrastructure

This project uses the [LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template)
v2.3.0 with BSD-3-Clause licensing. See [migration notes](https://github.com/ColinOrionChandler/CitSciAnalysis/blob/main/docs/migration.md) for
Copier updates, service activation status, and validation evidence. Versions come
from Git tags; no PyPI release is created by this migration.
