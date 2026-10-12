# Changelog

## Unreleased

- Deduplicate AA observations by object and datetime before selecting context,
  prefer pI over p2, and retain variant provenance without reassigning image scores.
  Version AA selections so old frozen worksets cannot bypass deduplication.

- Default AA context to ±3 and support metadata-only narrowing of saved worksets,
  retaining PNG-only publication, stable identities, and RCC defaults.

- Add default-on, resumable AA/RCC candidate worksets with configurable selection,
  training exclusions, hidden reveal CSVs, and a standalone preview command.
- Share AA images by object, merge archive context windows, validate and deduplicate
  PNG content, journal legacy migration, and bound remote retrieval to selected files.
- Preserve existing scoring outputs and support `--no-workset` for scoring alone.

- Optimize Rubin review-unit registry updates by using pandas grouped first-value
  aggregation while preserving the first non-null `object_id`, `subject_id`, and
  `batch` values across chunks.
- Add regression coverage for independent field completion and preservation of
  earlier registry values.
