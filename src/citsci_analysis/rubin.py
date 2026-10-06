"""Chunked scoring for Rubin Comet Catchers classifications.

Rubin subjects are scored at two grains. A review unit is the thing a
volunteer judged, currently one image and eventually an ordered pair or other
metadata-defined bundle. An object is the scientific target that can have many
review units. The scorer preserves both IDs so future paired-image workflows can
change the review-unit definition without changing the object ranking output.
"""

from __future__ import annotations

import math
import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .scoring import (
    DEFAULT_CUTOFF,
    DEFAULT_MAX_USER_CLASSIFICATIONS,
    DEFAULT_WEIGHTS,
    FALSE_STRINGS,
    TRUE_STRINGS,
    build_user_key,
    calculate_metric_weight,
    default_label,
    first_present_column,
    normalize_bool,
    normalize_thumbnames,
)


RUBIN_COLUMNS = [
    "classification_id",
    "user_name",
    "username",
    "user_id",
    "user_ip",
    "started_at",
    "finished_at",
    "subject_id",
    "review_unit_id",
    "pair_id",
    "object_id",
    "class",
    "thumbname",
    "filename",
    "metadata",
    "subject_data",
    "answer",
    "value",
    "train",
    "batch",
]


@dataclass(frozen=True)
class RubinScoringConfig:
    """Configuration for chunked Rubin Comet Catchers scoring."""

    metric_weights: tuple[float, float, float] = DEFAULT_WEIGHTS
    max_user_classifications: int = DEFAULT_MAX_USER_CLASSIFICATIONS
    min_user_classifications: int = 1
    cutoff: float = DEFAULT_CUTOFF
    chunk_size: int = 100_000
    dedupe_user_review_units: bool = True
    top_k: int = 3
    review_unit_column: Optional[str] = None
    object_column: Optional[str] = None
    problematic_training_review_units: frozenset[str] = frozenset()


@dataclass
class RubinScoreResult:
    """All score tables plus scalar diagnostics produced by Rubin scoring."""

    review_unit_scores: pd.DataFrame
    object_scores: pd.DataFrame
    user_weights: pd.DataFrame
    diagnostics: Mapping[str, float | int | str]


@dataclass(frozen=True)
class RubinProgressEvent:
    """Progress event emitted by streaming Rubin passes."""

    event: str
    phase: str
    rows: int = 0
    chunks: int = 0
    total_rows: Optional[int] = None


ProgressCallback = Callable[[RubinProgressEvent], None]


def _emit_progress(
    progress: Optional[ProgressCallback],
    event: str,
    phase: str,
    rows: int = 0,
    chunks: int = 0,
    total_rows: Optional[int] = None,
) -> None:
    if progress is not None:
        progress(RubinProgressEvent(event, phase, int(rows), int(chunks), total_rows))


def _existing_columns_for_parquet(path: Path, wanted: Iterable[str]) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Rubin Parquet scoring requires pyarrow. Install project requirements first.") from exc

    available = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in wanted if column in available]


def _existing_columns_for_csv(path: Path, wanted: Iterable[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    return [column for column in wanted if column in available]


def _existing_columns(path: Path, wanted: Iterable[str]) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return _existing_columns_for_parquet(path, wanted)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return _existing_columns_for_csv(path, wanted)
    raise ValueError(f"Unsupported Rubin input format: {path}")


def _rubin_input_row_count(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".parquet":
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Rubin Parquet scoring requires pyarrow. Install project requirements first.") from exc

    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_rubin_file_chunks(path: str | Path, config: RubinScoringConfig) -> Iterator[pd.DataFrame]:
    """Yield only the Rubin columns needed for scoring from CSV or Parquet."""

    input_path = Path(path)
    columns = _existing_columns(input_path, RUBIN_COLUMNS)
    suffixes = "".join(input_path.suffixes).lower()

    if input_path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(input_path)
        for batch in parquet_file.iter_batches(batch_size=config.chunk_size, columns=columns):
            yield batch.to_pandas()
        return

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        yield from pd.read_csv(input_path, usecols=columns, chunksize=config.chunk_size, low_memory=False)
        return

    raise ValueError(f"Unsupported Rubin input format: {input_path}")


def _normalize_answer_values(frame: pd.DataFrame) -> pd.Series:
    if "answer" in frame.columns:
        numeric = pd.to_numeric(frame["answer"], errors="coerce")
        answer = numeric.where(numeric.isin([0.0, 1.0]))

        raw = frame["answer"].astype("string").str.strip().str.lower()
        answer.loc[raw.isin(TRUE_STRINGS)] = 1.0
        answer.loc[raw.isin(FALSE_STRINGS)] = 0.0
        return answer.astype("float64")

    raw = first_present_column(frame, ["value"]).astype("string").str.strip().str.lower()
    answer = pd.Series(np.nan, index=frame.index, dtype="float64")
    answer.loc[raw.isin({"yes", "y", "true", "t", "1", "1.0"})] = 1.0
    answer.loc[raw.isin({"no", "n", "false", "f", "0", "0.0"})] = 0.0
    return answer


def _derive_object_ids(review_unit_ids: pd.Series) -> pd.Series:
    normalized = normalize_thumbnames(review_unit_ids)
    before_date = normalized.str.extract(r"^(.+?)_(?=\d{8}(?:[-_]|$))", expand=False)
    before_first_underscore = normalized.str.extract(r"^([^_]+)", expand=False)
    return before_date.combine_first(before_first_underscore).combine_first(normalized)


def _normalize_id_values(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.mask(text.str.lower().isin(FALSE_STRINGS))


def _fill_missing(base: pd.Series, fallback: pd.Series) -> pd.Series:
    return base.astype("string").mask(base.isna(), fallback.astype("string"))


def _is_missing_value(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _combined_present_columns(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    combined = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in names:
        if name not in frame.columns:
            continue
        combined = _fill_missing(combined, _normalize_id_values(frame[name]))
    return combined


def _metadata_lookup(payload: object, keys: Iterable[str]) -> object:
    """Find the first matching metadata value in nested subject payloads."""

    if _is_missing_value(payload):
        return pd.NA

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return pd.NA

    if not isinstance(payload, dict):
        return pd.NA

    key_options = []
    for key in keys:
        key_options.extend([key, f"#{key}"])

    stack = [payload]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        # Zooniverse exports may retain metadata names with or without a
        # leading "#"; paired-subject metadata can also be nested by image slot.
        for key in key_options:
            value = current.get(key)
            if value is not None and not _is_missing_value(value):
                return value
        stack.extend(value for value in current.values() if isinstance(value, dict))
    return pd.NA


def _metadata_values(frame: pd.DataFrame, keys: Iterable[str]) -> pd.Series:
    combined = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ("metadata", "subject_data"):
        if column not in frame.columns:
            continue
        values = frame[column].map(lambda payload: _metadata_lookup(payload, keys))
        combined = _fill_missing(combined, _normalize_id_values(values))
    return combined


def normalize_rubin_chunk(raw: pd.DataFrame, row_offset: int, config: RubinScoringConfig) -> pd.DataFrame:
    """Normalize a raw Rubin chunk into the schema used by both scoring passes.

    ID precedence is deliberate: explicit columns win, then metadata, then
    current-export fallbacks. That lets a future paired subject identify the
    review unit explicitly while current image-level exports still score.
    """

    review_candidates = [config.review_unit_column] if config.review_unit_column else []
    review_candidates += ["review_unit_id", "pair_id"]
    review_source = _combined_present_columns(raw, [name for name in review_candidates if name])
    review_source = _fill_missing(review_source, _metadata_values(raw, ["review_unit_id", "pair_id"]))
    review_source = _fill_missing(review_source, _combined_present_columns(raw, ["thumbname", "filename"]))
    review_unit_id = normalize_thumbnames(review_source)

    object_candidates = [config.object_column] if config.object_column else []
    object_candidates += ["object_id", "class"]
    object_id = _combined_present_columns(raw, [name for name in object_candidates if name])
    object_id = _fill_missing(object_id, _metadata_values(raw, ["object_id", "class"]))
    object_id = _fill_missing(object_id, _combined_present_columns(raw, ["subject_id"]))
    derived_object_id = _derive_object_ids(review_unit_id).astype("string")
    object_id = object_id.mask(object_id.isna(), derived_object_id)

    subject_id = _normalize_id_values(first_present_column(raw, ["subject_id"]))

    normalized = pd.DataFrame(
        {
            "row_order": np.arange(row_offset, row_offset + len(raw), dtype=np.int64),
            "classification_id": first_present_column(raw, ["classification_id"]),
            "user_key": build_user_key(raw),
            "user_name": first_present_column(raw, ["user_name", "username"]),
            "review_unit_id": review_unit_id,
            "object_id": object_id,
            "subject_id": subject_id,
            "answer": _normalize_answer_values(raw),
            "train": normalize_bool(first_present_column(raw, ["train"]), default=False),
            "batch": pd.to_numeric(first_present_column(raw, ["batch"]), errors="coerce"),
            "started_at": pd.to_datetime(first_present_column(raw, ["started_at"]), errors="coerce", utc=True),
            "finished_at": pd.to_datetime(first_present_column(raw, ["finished_at"]), errors="coerce", utc=True),
        }
    )
    for column in ("user_key", "user_name", "review_unit_id", "object_id", "subject_id"):
        normalized[column] = normalized[column].astype("string")
    return normalized


def _first_non_null(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _update_registry(registry: dict[str, dict[str, object]], chunk: pd.DataFrame) -> None:
    candidates = chunk.loc[chunk["review_unit_id"].notna(), ["review_unit_id", "object_id", "subject_id", "batch"]]
    if candidates.empty:
        return

    grouped = candidates.groupby("review_unit_id", sort=False, dropna=False).agg(
        object_id=("object_id", _first_non_null),
        subject_id=("subject_id", _first_non_null),
        batch=("batch", _first_non_null),
    )
    for review_unit_id, row in grouped.iterrows():
        key = str(review_unit_id)
        if key not in registry:
            registry[key] = {
                "review_unit_id": key,
                "object_id": row["object_id"],
                "subject_id": row["subject_id"],
                "batch": row["batch"],
            }
            continue
        current = registry[key]
        for column in ("object_id", "subject_id", "batch"):
            if pd.isna(current.get(column)) and pd.notna(row[column]):
                current[column] = row[column]


def _add_series_counts(target: dict[str, int], counts: pd.Series) -> None:
    for key, value in counts.items():
        if pd.isna(key):
            continue
        target[str(key)] = target.get(str(key), 0) + int(value)


def _add_grouped_stats(target: dict[str, dict[str, int]], stats: pd.DataFrame) -> None:
    for user_key, row in stats.iterrows():
        if pd.isna(user_key):
            continue
        user = str(user_key)
        current = target.setdefault(
            user,
            {"train_count": 0, "train_yes_count": 0, "test_count": 0, "test_yes_count": 0},
        )
        for column in current:
            current[column] += int(row[column])


def _prepare_chunk_factory(
    chunks: Callable[[], Iterable[pd.DataFrame]] | Iterable[pd.DataFrame],
) -> Callable[[], Iterable[pd.DataFrame]]:
    if callable(chunks):
        return chunks
    materialized = list(chunks)
    return lambda: iter(materialized)


def _first_pass(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    config: RubinScoringConfig,
    progress: Optional[ProgressCallback] = None,
    total_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, int | float]]:
    """Stream once to build volunteer weights and the review-unit registry."""

    user_seen: dict[str, int] = {}
    user_stats: dict[str, dict[str, int]] = {}
    registry: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, int | float] = {
        "rows": 0,
        "valid_answers": 0,
        "training_rows": 0,
    }

    phase = "Pass 1/2: user weights"
    _emit_progress(progress, "start", phase, total_rows=total_rows)

    row_offset = 0
    chunk_count = 0
    for raw in chunk_factory():
        chunk_count += 1
        chunk = normalize_rubin_chunk(raw, row_offset, config)
        row_offset += len(raw)
        diagnostics["rows"] += int(len(chunk))
        diagnostics["valid_answers"] += int(chunk["answer"].isin([0.0, 1.0]).sum())
        diagnostics["training_rows"] += int(chunk["train"].sum())
        _update_registry(registry, chunk)

        has_user = chunk["user_key"].notna()
        if has_user.any():
            # The per-user cap must be stable across chunk boundaries, so keep
            # the previous count and add each user's within-chunk order.
            prior = chunk.loc[has_user, "user_key"].astype(str).map(user_seen).fillna(0).astype("int64")
            within = chunk.loc[has_user].groupby("user_key", sort=False).cumcount() + 1
            user_index = pd.Series(pd.NA, index=chunk.index, dtype="Int64")
            user_index.loc[has_user] = (prior.to_numpy() + within.to_numpy()).astype("int64")
            _add_series_counts(user_seen, chunk.loc[has_user, "user_key"].astype(str).value_counts(sort=False))
        else:
            user_index = pd.Series(pd.NA, index=chunk.index, dtype="Int64")

        valid_answer = chunk["answer"].isin([0.0, 1.0])
        under_cap = user_index.le(config.max_user_classifications).fillna(False)
        problematic = chunk["train"] & chunk["review_unit_id"].astype("string").isin(config.problematic_training_review_units)
        stats_mask = valid_answer & under_cap & ~problematic & chunk["user_key"].notna()
        yes_mask = chunk["answer"] == 1.0
        train_mask = stats_mask & chunk["train"]
        test_mask = stats_mask & ~chunk["train"]

        stats = pd.DataFrame(
            {
                "user_key": chunk["user_key"],
                "train_count": train_mask.astype("int64"),
                "train_yes_count": (train_mask & yes_mask).astype("int64"),
                "test_count": test_mask.astype("int64"),
                "test_yes_count": (test_mask & yes_mask).astype("int64"),
            }
        )
        grouped = stats.loc[stats_mask].groupby("user_key", sort=False, dropna=False).sum()
        _add_grouped_stats(user_stats, grouped)
        _emit_progress(progress, "update", phase, row_offset, chunk_count, total_rows)

    _emit_progress(progress, "finish", phase, row_offset, chunk_count, total_rows)
    _emit_progress(progress, "message", "Calculating user weights...")

    all_users = sorted(set(user_seen) | set(user_stats))
    user_weights = pd.DataFrame(index=pd.Index(all_users, name="user_key"))
    for column in ("train_count", "train_yes_count", "test_count", "test_yes_count"):
        user_weights[column] = [user_stats.get(user, {}).get(column, 0) for user in all_users]
    user_weights["classification_count"] = [user_seen.get(user, 0) for user in all_users]
    user_weights["weight"] = calculate_metric_weight(
        user_weights["train_yes_count"],
        user_weights["train_count"],
        user_weights["test_yes_count"],
        user_weights["test_count"],
        config.metric_weights,
    )
    _emit_progress(progress, "message", f"Calculated weights for {len(user_weights):,} users.")
    return user_weights.reset_index(), registry, diagnostics


def _prepare_votes(chunk: pd.DataFrame, user_weights: pd.DataFrame, config: RubinScoringConfig) -> pd.DataFrame:
    """Attach final volunteer weights to valid review-unit votes."""

    user_lookup = user_weights.set_index("user_key")
    weights = user_lookup["weight"]
    classification_counts = user_lookup["classification_count"]

    valid = chunk["review_unit_id"].notna() & chunk["answer"].isin([0.0, 1.0]) & chunk["user_key"].notna()
    if config.min_user_classifications > 1:
        valid &= chunk["user_key"].map(classification_counts).fillna(0) >= config.min_user_classifications

    votes = chunk.loc[valid, ["row_order", "review_unit_id", "object_id", "subject_id", "batch", "user_key", "answer"]].copy()
    if votes.empty:
        return votes
    votes["weight"] = votes["user_key"].map(weights).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    return votes


def _sqlite_vote_table(connection: sqlite3.Connection, dedupe: bool) -> None:
    """Create the temporary vote table used for cross-chunk aggregation."""

    connection.execute("DROP TABLE IF EXISTS votes")
    if dedupe:
        connection.execute(
            """
            CREATE TABLE votes (
                review_unit_id TEXT NOT NULL,
                user_key TEXT NOT NULL,
                row_order INTEGER NOT NULL,
                object_id TEXT,
                subject_id TEXT,
                batch REAL,
                answer REAL NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (review_unit_id, user_key)
            )
            """
        )
    else:
        connection.execute(
            """
            CREATE TABLE votes (
                review_unit_id TEXT NOT NULL,
                user_key TEXT NOT NULL,
                row_order INTEGER NOT NULL,
                object_id TEXT,
                subject_id TEXT,
                batch REAL,
                answer REAL NOT NULL,
                weight REAL NOT NULL
            )
            """
        )


def _insert_votes(connection: sqlite3.Connection, votes: pd.DataFrame, dedupe: bool) -> None:
    """Insert scored votes, optionally keeping the latest user vote per review unit."""

    if votes.empty:
        return
    columns = ["review_unit_id", "user_key", "row_order", "object_id", "subject_id", "batch", "answer", "weight"]
    records = []
    for row in votes[columns].itertuples(index=False, name=None):
        records.append(tuple(None if pd.isna(value) else value for value in row))
    if dedupe:
        connection.executemany(
            """
            INSERT INTO votes (
                review_unit_id, user_key, row_order, object_id, subject_id, batch, answer, weight
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(review_unit_id, user_key) DO UPDATE SET
                row_order = excluded.row_order,
                object_id = excluded.object_id,
                subject_id = excluded.subject_id,
                batch = excluded.batch,
                answer = excluded.answer,
                weight = excluded.weight
            WHERE excluded.row_order >= votes.row_order
            """,
            records,
        )
    else:
        connection.executemany(
            """
            INSERT INTO votes (
                review_unit_id, user_key, row_order, object_id, subject_id, batch, answer, weight
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def _aggregate_votes(connection: sqlite3.Connection) -> pd.DataFrame:
    """Collapse the temporary vote table into one weighted score row per review unit."""

    return pd.read_sql_query(
        """
        SELECT
            review_unit_id,
            MIN(object_id) AS object_id,
            MIN(subject_id) AS subject_id,
            MIN(batch) AS batch,
            SUM(weight * answer) AS numerator,
            SUM(weight) AS denominator,
            COUNT(*) AS vote_count,
            SUM(answer) AS yes_count
        FROM votes
        GROUP BY review_unit_id
        """,
        connection,
    )


def _score_review_units(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    user_weights: pd.DataFrame,
    registry: dict[str, dict[str, object]],
    config: RubinScoringConfig,
    progress: Optional[ProgressCallback] = None,
    total_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int]:
    """Stream a second time to score review units with final user weights."""

    phase = "Pass 2/2: score review units"
    _emit_progress(progress, "start", phase, total_rows=total_rows)

    with tempfile.TemporaryDirectory(prefix="rubin_votes_") as temporary_directory:
        database_path = Path(temporary_directory) / "votes.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            _sqlite_vote_table(connection, config.dedupe_user_review_units)
            row_offset = 0
            chunk_count = 0
            for raw in chunk_factory():
                chunk_count += 1
                chunk = normalize_rubin_chunk(raw, row_offset, config)
                row_offset += len(raw)
                votes = _prepare_votes(chunk, user_weights, config)
                _insert_votes(connection, votes, config.dedupe_user_review_units)
                connection.commit()
                _emit_progress(progress, "update", phase, row_offset, chunk_count, total_rows)

            _emit_progress(progress, "finish", phase, row_offset, chunk_count, total_rows)
            _emit_progress(progress, "message", "Aggregating review unit votes...")
            scored_votes = int(connection.execute("SELECT COUNT(*) FROM votes").fetchone()[0])
            grouped = _aggregate_votes(connection)
            _emit_progress(progress, "message", f"Aggregated {scored_votes:,} scored votes.")
        finally:
            connection.close()

    registry_frame = pd.DataFrame(registry.values())
    if registry_frame.empty:
        registry_frame = pd.DataFrame(columns=["review_unit_id", "object_id", "subject_id", "batch"])
    _emit_progress(progress, "message", "Finalizing review unit scores...")
    review_units = registry_frame.merge(grouped, on="review_unit_id", how="left", suffixes=("", "_scored"))
    for column in ("object_id", "subject_id", "batch"):
        scored_column = f"{column}_scored"
        if scored_column in review_units.columns:
            review_units[column] = review_units[column].mask(review_units[column].isna(), review_units[scored_column])
            review_units = review_units.drop(columns=[scored_column])

    for column in ("numerator", "denominator", "vote_count", "yes_count"):
        review_units[column] = review_units[column].fillna(0)
    denominator = review_units["denominator"].to_numpy(dtype="float64")
    score = np.zeros(len(review_units), dtype="float64")
    np.divide(
        review_units["numerator"].to_numpy(dtype="float64"),
        denominator,
        out=score,
        where=denominator != 0,
    )
    review_units["score"] = score
    review_units["vote_count"] = review_units["vote_count"].astype("int64")
    review_units["yes_count"] = review_units["yes_count"].astype("int64")
    review_units["weight_sum"] = review_units["denominator"].astype("float64")
    return review_units, scored_votes


def _top_k_mean(values: pd.Series, top_k: int) -> float:
    if values.empty:
        return 0.0
    return float(values.nlargest(max(1, top_k)).mean())


def _build_object_scores(review_units: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Roll review-unit scores up to object rankings.

    The primary object score is the top-k mean, which rewards objects with more
    than one strong review unit without letting a large number of low-confidence
    images swamp the ranking.
    """

    if review_units.empty:
        return pd.DataFrame(
            columns=[
                "object_id",
                "object_score",
                "mean_review_unit_score",
                "peak_review_unit_score",
                "top_3_mean_score",
                "review_unit_count",
                "vote_count",
                "best_review_unit",
            ]
        )

    grouped = review_units.groupby("object_id", sort=False, dropna=False)
    object_scores = grouped.agg(
        mean_review_unit_score=("score", "mean"),
        peak_review_unit_score=("score", "max"),
        review_unit_count=("review_unit_id", "size"),
        vote_count=("vote_count", "sum"),
    ).reset_index()
    top_scores = grouped["score"].apply(lambda values: _top_k_mean(values, top_k)).rename("top_3_mean_score").reset_index()
    object_scores = object_scores.merge(top_scores, on="object_id", how="left")
    object_scores["object_score"] = object_scores["top_3_mean_score"]

    best = (
        review_units.sort_values(["object_id", "score", "vote_count", "review_unit_id"], ascending=[True, False, False, True])
        .drop_duplicates("object_id", keep="first")[["object_id", "review_unit_id"]]
        .rename(columns={"review_unit_id": "best_review_unit"})
    )
    object_scores = object_scores.merge(best, on="object_id", how="left")
    return object_scores[
        [
            "object_id",
            "object_score",
            "mean_review_unit_score",
            "peak_review_unit_score",
            "top_3_mean_score",
            "review_unit_count",
            "vote_count",
            "best_review_unit",
        ]
    ]


def score_rubin_chunks(
    chunks: Callable[[], Iterable[pd.DataFrame]] | Iterable[pd.DataFrame],
    config: RubinScoringConfig | None = None,
    progress: Optional[ProgressCallback] = None,
    total_rows: Optional[int] = None,
) -> RubinScoreResult:
    """Score Rubin classifications from a reusable chunk source."""

    active_config = config or RubinScoringConfig()
    chunk_factory = _prepare_chunk_factory(chunks)
    user_weights, registry, diagnostics = _first_pass(chunk_factory, active_config, progress, total_rows)
    review_units, scored_votes = _score_review_units(
        chunk_factory,
        user_weights,
        registry,
        active_config,
        progress,
        total_rows,
    )
    _emit_progress(progress, "message", "Building object scores...")
    object_scores = _build_object_scores(review_units, active_config.top_k)
    _emit_progress(progress, "message", f"Built {len(object_scores):,} object scores.")

    diagnostics.update(
        {
            "mode": "rubin",
            "review_units": int(review_units["review_unit_id"].nunique(dropna=True)),
            "objects": int(object_scores["object_id"].nunique(dropna=True)),
            "scored_votes": scored_votes,
            "users": int(user_weights["user_key"].nunique(dropna=True)),
            "object_score_min": float(object_scores["object_score"].min()) if not object_scores.empty else math.nan,
            "object_score_max": float(object_scores["object_score"].max()) if not object_scores.empty else math.nan,
            "object_score_mean": float(object_scores["object_score"].mean()) if not object_scores.empty else math.nan,
            "objects_above_cutoff": int((object_scores["object_score"] >= active_config.cutoff).sum()),
        }
    )
    return RubinScoreResult(
        review_unit_scores=review_units,
        object_scores=object_scores,
        user_weights=user_weights,
        diagnostics=diagnostics,
    )


def score_rubin_file(
    path: str | Path,
    config: RubinScoringConfig | None = None,
    progress: Optional[ProgressCallback] = None,
) -> RubinScoreResult:
    """Score a CSV or Parquet Rubin export from disk."""

    active_config = config or RubinScoringConfig()
    total_rows = _rubin_input_row_count(Path(path))
    return score_rubin_chunks(
        lambda: iter_rubin_file_chunks(path, active_config),
        active_config,
        progress,
        total_rows,
    )


def write_rubin_outputs(
    result: RubinScoreResult, output_dir: str | Path, label: Optional[str], cutoff: float
) -> Mapping[str, Path]:
    """Write Rubin review-unit, object, and volunteer-weight CSV outputs."""

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_label = label or default_label()

    review_unit_path = outdir / f"rubin_review_unit_scores_{output_label}.csv"
    object_path = outdir / f"rubin_object_scores_{output_label}.csv"
    weights_path = outdir / f"rubin_user_weights_{output_label}.csv"

    review_columns = ["review_unit_id", "object_id", "subject_id", "score", "vote_count", "yes_count", "weight_sum", "batch"]
    object_columns = [
        "object_id",
        "object_score",
        "mean_review_unit_score",
        "peak_review_unit_score",
        "top_3_mean_score",
        "review_unit_count",
        "vote_count",
        "best_review_unit",
    ]
    result.review_unit_scores[review_columns].to_csv(review_unit_path, index=False)
    result.object_scores[object_columns].sort_values("object_score", ascending=False).to_csv(object_path, index=False)
    result.user_weights.to_csv(weights_path, index=False)
    return {"review_units": review_unit_path, "objects": object_path, "user_weights": weights_path}
