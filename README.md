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

Install the dependencies in your environment:

```bash
python3 -m pip install -r requirements.txt
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

- `output_total_<label>.csv`: every normalized `thumbname` and `score`.
- `output_<label>.csv`: only rows at or above the cutoff, default `0.473`.
- `user_weights_<label>.csv`: volunteer weight diagnostics.

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

## Development Checks

Run the test suite from an environment with the project dependencies installed:

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
