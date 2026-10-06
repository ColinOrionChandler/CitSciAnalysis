# Changelog

## Unreleased

- Optimize Rubin review-unit registry updates by using pandas grouped first-value
  aggregation while preserving the first non-null `object_id`, `subject_id`, and
  `batch` values across chunks.
- Add regression coverage for independent field completion and preservation of
  earlier registry values.
