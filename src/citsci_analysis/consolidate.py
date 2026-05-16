from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from .scoring import FALSE_STRINGS
except ImportError:
    if __package__ not in {None, ""}:
        raise
    FALSE_STRINGS = {"", "0", "0.0", "false", "f", "no", "n", "none", "nan", "null", "na", "n/a", "<na>"}


THUMBNAIL_COLUMNS = ("thumbnail_basename", "thumbname", "thumb_name", "ThumbName", "thumbnail", "filename")
PATH_COLUMNS = ("path", "filepath", "file_path", "image_path")
SCORE_COLUMNS = ("score", "Confidence", "confidence")
DATETIME_COLUMNS = ("datetime", "date", "timestamp", "scored_at", "created_at")
OUTPUT_COLUMNS = ["datetime", "path", "thumbnail_basename", "score"]


@dataclass(frozen=True)
class ConsolidationResult:
    """Normalized consolidated score rows plus scalar diagnostics."""

    scores: pd.DataFrame
    summary: dict[str, int | str]


@dataclass(frozen=True)
class _ParsedRow:
    datetime: str
    parsed_datetime: Optional[datetime]
    has_datetime: bool
    malformed_datetime: bool
    path: str
    thumbnail_basename: str
    score: float
    input_order: int


def discover_score_csvs(directory: str | Path, output_path: str | Path | None = None) -> list[Path]:
    """Return recursive CSV inputs in deterministic order, excluding output."""

    root = Path(directory)
    output_resolved = Path(output_path).resolve() if output_path is not None else None
    csv_paths = []
    for path in root.rglob("*.csv"):
        if not path.is_file():
            continue
        if output_resolved is not None and path.resolve() == output_resolved:
            continue
        csv_paths.append(path)
    return sorted(csv_paths, key=lambda value: str(value.resolve()))


def consolidate_score_directory(
    directory: str | Path,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ConsolidationResult:
    """Read score CSVs recursively, dedupe by thumbnail basename, and write output."""

    root = Path(directory)
    output = Path(output_path) if output_path is not None else root / "consolidated_scores.csv"
    csv_paths = discover_score_csvs(root, output)

    if progress is not None:
        progress(f"Found {len(csv_paths):,} CSV files.")

    parsed_rows: list[_ParsedRow] = []
    input_rows = 0
    skipped_rows = 0
    next_input_order = 0

    for file_index, csv_path in enumerate(csv_paths, start=1):
        if progress is not None:
            progress(f"Parsing CSV {file_index:,}/{len(csv_paths):,}: {csv_path}")

        file_rows, file_input_rows, file_skipped = _parse_score_csv(csv_path, next_input_order)
        parsed_rows.extend(file_rows)
        input_rows += file_input_rows
        skipped_rows += file_skipped
        next_input_order += file_input_rows

    normalized = _rows_to_frame(parsed_rows)
    consolidated = _dedupe_latest(normalized)

    output.parent.mkdir(parents=True, exist_ok=True)
    consolidated[OUTPUT_COLUMNS].to_csv(output, index=False)

    summary: dict[str, int | str] = {
        "discovered_files": len(csv_paths),
        "input_rows": input_rows,
        "parsed_rows": len(parsed_rows),
        "skipped_rows": skipped_rows,
        "malformed_date_rows": int(sum(row.malformed_datetime for row in parsed_rows)),
        "unique_basenames": int(consolidated["thumbnail_basename"].nunique()) if not consolidated.empty else 0,
        "duplicate_rows_removed": len(parsed_rows) - len(consolidated),
        "output_path": str(output),
    }
    return ConsolidationResult(scores=consolidated[OUTPUT_COLUMNS], summary=summary)


def _parse_score_csv(path: Path, input_order_start: int) -> tuple[list[_ParsedRow], int, int]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0, 0

        if _looks_like_headered_score_file(header):
            return _parse_headered_rows(header, reader, input_order_start)
        return _parse_headerless_rows(_prepend_row(header, reader), input_order_start)


def _prepend_row(first_row: list[str], rows: Iterable[list[str]]) -> Iterable[list[str]]:
    yield first_row
    yield from rows


def _parse_headerless_rows(rows: Iterable[list[str]], input_order_start: int) -> tuple[list[_ParsedRow], int, int]:
    parsed: list[_ParsedRow] = []
    input_rows = 0
    skipped = 0
    for row_number, row in enumerate(rows):
        input_rows += 1
        input_order = input_order_start + row_number
        if len(row) < 4:
            skipped += 1
            continue
        parsed_row = _build_row(
            datetime_value=row[0],
            path_value=row[2],
            thumbnail_value=row[2],
            score_value=row[3],
            input_order=input_order,
        )
        if parsed_row is None:
            skipped += 1
            continue
        parsed.append(parsed_row)
    return parsed, input_rows, skipped


def _parse_headered_rows(
    header: list[str],
    rows: Iterable[list[str]],
    input_order_start: int,
) -> tuple[list[_ParsedRow], int, int]:
    parsed: list[_ParsedRow] = []
    input_rows = 0
    skipped = 0

    thumb_index = _find_column(header, THUMBNAIL_COLUMNS)
    path_index = _find_column(header, PATH_COLUMNS)
    score_index = _find_column(header, SCORE_COLUMNS)
    datetime_index = _find_column(header, DATETIME_COLUMNS)

    for row_number, row in enumerate(rows):
        input_rows += 1
        input_order = input_order_start + row_number
        path_value = _get_cell(row, path_index)
        thumbnail_value = _get_cell(row, thumb_index) or path_value
        parsed_row = _build_row(
            datetime_value=_get_cell(row, datetime_index),
            path_value=path_value,
            thumbnail_value=thumbnail_value,
            score_value=_get_cell(row, score_index),
            input_order=input_order,
        )
        if parsed_row is None:
            skipped += 1
            continue
        parsed.append(parsed_row)
    return parsed, input_rows, skipped


def _looks_like_headered_score_file(header: Iterable[str]) -> bool:
    header_list = list(header)
    score_index = _find_column(header_list, SCORE_COLUMNS)
    if score_index is None:
        return False
    return _find_column(header_list, THUMBNAIL_COLUMNS) is not None or _find_column(header_list, PATH_COLUMNS) is not None


def _build_row(
    *,
    datetime_value: object,
    path_value: object,
    thumbnail_value: object,
    score_value: object,
    input_order: int,
) -> _ParsedRow | None:
    thumbnail = _normalize_one_thumbnail(thumbnail_value)
    if thumbnail == "":
        return None

    score = _parse_score(score_value)
    if score is None:
        return None

    parsed_datetime, formatted_datetime, had_datetime, malformed_datetime = _parse_datetime(datetime_value)
    return _ParsedRow(
        datetime=formatted_datetime,
        parsed_datetime=parsed_datetime,
        has_datetime=had_datetime and parsed_datetime is not None,
        malformed_datetime=malformed_datetime,
        path=_clean_text(path_value),
        thumbnail_basename=thumbnail,
        score=score,
        input_order=input_order,
    )


def _rows_to_frame(rows: list[_ParsedRow]) -> pd.DataFrame:
    pd = _pandas()
    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "path",
                "thumbnail_basename",
                "score",
                "_parsed_datetime",
                "_has_datetime",
                "_input_order",
            ]
        )

    return pd.DataFrame(
        {
            "datetime": [row.datetime for row in rows],
            "path": [row.path for row in rows],
            "thumbnail_basename": [row.thumbnail_basename for row in rows],
            "score": [row.score for row in rows],
            "_parsed_datetime": [row.parsed_datetime for row in rows],
            "_has_datetime": [row.has_datetime for row in rows],
            "_input_order": [row.input_order for row in rows],
        }
    )


def _dedupe_latest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ordered = frame.sort_values(
        ["thumbnail_basename", "_has_datetime", "_parsed_datetime", "_input_order"],
        ascending=[True, True, True, True],
        na_position="first",
        kind="mergesort",
    )
    deduped = ordered.groupby("thumbnail_basename", sort=False, as_index=False).tail(1)
    return deduped.sort_values("thumbnail_basename", kind="mergesort").reset_index(drop=True)


def _find_column(header: list[str], candidates: Iterable[str]) -> int | None:
    normalized_header = {_header_key(value): index for index, value in enumerate(header)}
    for candidate in candidates:
        index = normalized_header.get(_header_key(candidate))
        if index is not None:
            return index
    return None


def _header_key(value: object) -> str:
    return "".join(str(value).strip().lower().split()).replace("_", "")


def _get_cell(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index]


def _normalize_one_thumbnail(value: object) -> str:
    cleaned = _clean_text(value)
    if cleaned == "":
        return ""
    normalized = cleaned.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return _strip_thumbnail_extension(normalized)


def _strip_thumbnail_extension(value: str) -> str:
    return re.sub(r"(?i)\.(?:png|jpe?g|gif|tiff?|webp|bmp|fits(?:\.fz)?|fit(?:\.fz)?|fz)$", "", value)


def _parse_score(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_datetime(value: object) -> tuple[Optional[datetime], str, bool, bool]:
    text = _clean_text(value)
    if text == "":
        return None, "", False, False

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, "", True, True

    return parsed, parsed.isoformat(sep=" "), True, False


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in FALSE_STRINGS:
        return ""
    return text


def _pandas():
    import pandas as pd

    return pd


def _progress_enabled(setting: str, stream: object) -> bool:
    if setting == "always":
        return True
    if setting == "never":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate recursive score CSVs to the latest score per thumbnail basename."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory to search recursively for CSV files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to consolidated_scores.csv in the input directory.",
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "always", "never"],
        default="auto",
        help="Show consolidation progress on stderr. Defaults to auto in interactive terminals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    show_progress = _progress_enabled(args.progress, sys.stderr)
    progress = (lambda text: print(text, file=sys.stderr, flush=True)) if show_progress else None
    result = consolidate_score_directory(args.directory, args.output, progress)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
