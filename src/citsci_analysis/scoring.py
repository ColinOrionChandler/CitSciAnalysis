from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


DEFAULT_WEIGHTS = (7.0, 2.0, 1.0)
DEFAULT_CUTOFF = 0.473
DEFAULT_MAX_USER_CLASSIFICATIONS = 10_000
DEFAULT_COC_SCORES_PATH = Path(
    "/Users/colinchandler/Dropbox/School/Research/Thumbnail Examination UI/consolidated_scores.csv"
)

TRUE_STRINGS = {"1", "1.0", "true", "t", "yes", "y"}
FALSE_STRINGS = {"", "0", "0.0", "false", "f", "no", "n", "none", "nan", "null", "na", "n/a", "<na>"}

NORMALIZED_COLUMNS = [
    "row_order",
    "classification_id",
    "user_key",
    "user_name",
    "thumbname",
    "answer",
    "train",
    "class",
    "batch",
    "started_at",
    "finished_at",
]

THUMBNAIL_SUMMARY_COLUMNS = [
    "number_of_times_classified",
    "number_of_yes_answers",
    "number_of_no_answers",
    "fraction_yes",
    "train",
    "class",
]

SCORE_OUTPUT_COLUMNS = ["thumbname", "score", "coc_score", *THUMBNAIL_SUMMARY_COLUMNS]
COC_THUMBNAIL_COLUMNS = ("thumbnail_basename", "thumbname", "thumb_name", "ThumbName", "thumbnail", "filename")
COC_PATH_COLUMNS = ("path", "filepath", "file_path", "image_path")
COC_SCORE_COLUMNS = ("score", "Confidence", "confidence")

PARQUET_COLUMNS = [
    "classification_id",
    "user_name",
    "user_id",
    "user_ip",
    "started_at",
    "finished_at",
    "thumbname",
    "filename",
    "answer",
    "train",
    "class",
    "batch",
]

CSV_COLUMNS = [
    "classification_id",
    "username",
    "user_name",
    "user_id",
    "user_ip",
    "started_at",
    "finished_at",
    "thumbname",
    "filename",
    "value",
    "answer",
    "train",
    "class",
    "batch",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for user weighting and thumbnail scoring."""

    mode: str = "naive"
    metric_weights: tuple[float, float, float] = DEFAULT_WEIGHTS
    max_user_classifications: int = DEFAULT_MAX_USER_CLASSIFICATIONS
    min_user_classifications: int = 1
    max_votes_per_thumb: Optional[int] = None
    dedupe_user_thumbnail: bool = True
    score_all_same_shortcut: bool = False
    cutoff: float = DEFAULT_CUTOFF
    problematic_training_thumbs: frozenset[str] = frozenset()


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    user_weights: pd.DataFrame
    diagnostics: Mapping[str, float | int | str]


def _existing_columns_for_csv(path: Path, wanted: Iterable[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    return [column for column in wanted if column in available]


def _existing_columns_for_parquet(path: Path, wanted: Iterable[str]) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return list(wanted)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in wanted if column in available]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported classification table with only the columns we need."""

    input_path = Path(path)
    suffixes = "".join(input_path.suffixes).lower()
    if input_path.suffix.lower() == ".parquet":
        columns = _existing_columns_for_parquet(input_path, PARQUET_COLUMNS)
        return pd.read_parquet(input_path, columns=columns)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        columns = _existing_columns_for_csv(input_path, CSV_COLUMNS)
        return pd.read_csv(input_path, usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported input format: {input_path}")


def first_present_column(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    """Return the first present column, or an all-null series if none exist."""

    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(pd.NA, index=frame.index)


def coalesce_present_columns(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    """Return the first non-empty value per row from the requested columns."""

    combined = pd.Series(pd.NA, index=frame.index, dtype="string")
    for name in names:
        if name not in frame.columns:
            continue
        candidate = frame[name].astype("string").str.strip()
        candidate = candidate.mask(candidate.str.lower().isin(FALSE_STRINGS))
        combined = combined.combine_first(candidate)
    return combined


def normalize_thumbnames(values: pd.Series) -> pd.Series:
    """Normalize thumbnails to basename without image/fits suffixes."""

    text = values.astype("string").str.strip()
    text = text.mask(text.str.lower().isin(FALSE_STRINGS))
    text = text.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    return text.str.replace(r"(?i)\.(?:png|fits(?:\.fz)?|fit(?:\.fz)?|fz)$", "", regex=True)


def normalize_answers(frame: pd.DataFrame) -> pd.Series:
    """Normalize old yes/no values or cleaned numeric answers to 0/1 floats."""

    if "answer" in frame.columns:
        answer = pd.to_numeric(frame["answer"], errors="coerce")
        return answer.where(answer.isin([0.0, 1.0]))

    raw = first_present_column(frame, ["value"]).astype("string").str.strip().str.lower()
    answer = pd.Series(np.nan, index=frame.index, dtype="float64")
    answer.loc[raw == "yes"] = 1.0
    answer.loc[raw == "no"] = 0.0
    return answer


def normalize_bool(values: pd.Series, *, default: bool = False) -> pd.Series:
    """Normalize bool-like export values while preserving a caller-supplied default."""

    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series(default, index=values.index, dtype="bool")
    result.loc[text.isin(TRUE_STRINGS)] = True
    result.loc[text.isin(FALSE_STRINGS)] = False
    return result


def normalize_optional_text(values: pd.Series) -> pd.Series:
    """Normalize optional text metadata while preserving meaningful values."""

    text = values.astype("string").str.strip()
    missing_strings = {"", "none", "nan", "null", "<na>"}
    return text.mask(text.str.lower().isin(missing_strings))


def _normalize_identifier_values(values: pd.Series) -> pd.Series:
    """Stringify IDs while avoiding CSV-inferred integer values like ``42.0``."""

    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    integerish_float_text = text.str.match(r"^-?\d+\.0+$").fillna(False)
    integral = numeric.notna() & np.isclose(numeric % 1, 0) & integerish_float_text
    if integral.any():
        text.loc[integral] = numeric.loc[integral].astype("Int64").astype("string")
    return text.mask(text.str.lower().isin(FALSE_STRINGS))


def build_user_key(frame: pd.DataFrame) -> pd.Series:
    """Build a stable volunteer key from name, then ID/IP/classification fallback."""

    user_name = first_present_column(frame, ["user_name", "username"]).astype("string").str.strip()
    user_name = user_name.mask(user_name.str.lower().isin(FALSE_STRINGS))

    fallback = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column, prefix in (("user_id", "user_id:"), ("user_ip", "user_ip:"), ("classification_id", "classification:")):
        if column not in frame.columns:
            continue
        candidate = _normalize_identifier_values(frame[column])
        candidate = prefix + candidate
        fallback = fallback.combine_first(candidate)

    return user_name.combine_first(fallback)


def load_classifications(path: str | Path) -> pd.DataFrame:
    """Load and normalize classifications from old CSV or new cleaned Parquet."""

    raw = read_table(path)
    thumb_source = coalesce_present_columns(raw, ["thumbname", "filename"])
    normalized = pd.DataFrame(
        {
            "row_order": np.arange(len(raw), dtype=np.int64),
            "classification_id": first_present_column(raw, ["classification_id"]),
            "user_key": build_user_key(raw),
            "user_name": first_present_column(raw, ["user_name", "username"]),
            "thumbname": normalize_thumbnames(thumb_source),
            "answer": normalize_answers(raw),
            "train": normalize_bool(first_present_column(raw, ["train"]), default=False),
            "class": normalize_optional_text(first_present_column(raw, ["class"])),
            "batch": pd.to_numeric(first_present_column(raw, ["batch"]), errors="coerce"),
            "started_at": pd.to_datetime(first_present_column(raw, ["started_at"]), errors="coerce", utc=True),
            "finished_at": pd.to_datetime(first_present_column(raw, ["finished_at"]), errors="coerce", utc=True),
        }
    )
    normalized["user_key"] = normalized["user_key"].astype("string")
    normalized["user_name"] = normalized["user_name"].astype("string")
    normalized["thumbname"] = normalized["thumbname"].astype("string")
    normalized["class"] = normalized["class"].astype("string")
    return normalized[NORMALIZED_COLUMNS]


def load_problematic_training_thumbs(path: str | Path) -> frozenset[str]:
    """Load historic problematic-training metadata from pickle or CSV."""

    input_path = Path(path)
    if input_path.suffix.lower() in {".pkl", ".pickle"}:
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(input_path, low_memory=False)

    thumb_column = next((name for name in ("thumb_name", "thumbname", "Thumbname", "filename") if name in frame.columns), None)
    if thumb_column is None:
        raise ValueError(f"No thumbnail column found in problematic-training file: {input_path}")

    problem_column = next((name for name in frame.columns if "problematic" in str(name).lower()), None)
    if problem_column is None:
        raise ValueError(f"No problematic flag column found in problematic-training file: {input_path}")

    problem_text = frame[problem_column].astype("string").str.strip().str.lower()
    is_problematic = ~problem_text.isin({"no", "false", "0", "0.0"})
    thumbs = normalize_thumbnames(frame.loc[is_problematic, thumb_column]).dropna()
    return frozenset(thumbs.astype(str))


def _ensure_user_index(frame: pd.DataFrame) -> pd.DataFrame:
    if "_user_index" in frame.columns:
        return frame
    work = frame.copy()
    work["_user_index"] = work.groupby("user_key", sort=False).cumcount() + 1
    return work


def calculate_metric_weight(
    train_yes_count: pd.Series | np.ndarray,
    train_count: pd.Series | np.ndarray,
    test_yes_count: pd.Series | np.ndarray,
    test_count: pd.Series | np.ndarray,
    metric_weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Calculate the three-part volunteer weight with missing terms omitted."""

    a, b, c = metric_weights
    train_yes = np.asarray(train_yes_count, dtype="float64")
    train_total = np.asarray(train_count, dtype="float64")
    test_yes = np.asarray(test_yes_count, dtype="float64")
    test_total = np.asarray(test_count, dtype="float64")
    total = train_total + test_total

    numerator = np.zeros_like(total, dtype="float64")
    denominator = np.zeros_like(total, dtype="float64")

    train_mask = train_total > 0
    numerator[train_mask] += a * (train_yes[train_mask] / train_total[train_mask])
    denominator[train_mask] += a

    experience_mask = total > 0
    numerator[experience_mask] += b * (np.log10(total[experience_mask]) / 4.0)
    denominator[experience_mask] += b

    test_mask = test_total > 0
    numerator[test_mask] += c * (1.0 - (test_yes[test_mask] / test_total[test_mask]))
    denominator[test_mask] += c

    weights = np.zeros_like(total, dtype="float64")
    np.divide(numerator, denominator, out=weights, where=denominator != 0)
    return weights


def _stat_columns(frame: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    work = _ensure_user_index(frame)
    valid_answer = work["answer"].isin([0.0, 1.0])
    under_cap = work["_user_index"] <= config.max_user_classifications
    problematic = work["train"] & work["thumbname"].astype("string").isin(config.problematic_training_thumbs)
    stats_mask = valid_answer & under_cap & ~problematic

    train_mask = stats_mask & work["train"]
    test_mask = stats_mask & ~work["train"]
    yes_mask = work["answer"] == 1.0

    stats = pd.DataFrame(
        {
            "user_key": work["user_key"],
            "train_count": train_mask.astype("int64"),
            "train_yes_count": (train_mask & yes_mask).astype("int64"),
            "test_count": test_mask.astype("int64"),
            "test_yes_count": (test_mask & yes_mask).astype("int64"),
        },
        index=work.index,
    )
    return stats


def compute_static_user_weights(frame: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    """Compute one final volunteer weight per user for static scoring."""

    stats = _stat_columns(frame, config)
    grouped = stats.groupby("user_key", sort=False, dropna=False).agg(
        train_count=("train_count", "sum"),
        train_yes_count=("train_yes_count", "sum"),
        test_count=("test_count", "sum"),
        test_yes_count=("test_yes_count", "sum"),
    )
    grouped["classification_count"] = frame.groupby("user_key", sort=False, dropna=False).size()
    grouped["weight"] = calculate_metric_weight(
        grouped["train_yes_count"],
        grouped["train_count"],
        grouped["test_yes_count"],
        grouped["test_count"],
        config.metric_weights,
    )
    return grouped.reset_index()


def compute_time_row_weights(frame: pd.DataFrame, config: ScoringConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Compute a cumulative volunteer weight at each classification row."""

    stats = _stat_columns(frame, config)
    grouped_stats = stats.groupby("user_key", sort=False, dropna=False)
    cumulative = grouped_stats[["train_count", "train_yes_count", "test_count", "test_yes_count"]].cumsum()
    row_weights = pd.Series(
        calculate_metric_weight(
            cumulative["train_yes_count"],
            cumulative["train_count"],
            cumulative["test_yes_count"],
            cumulative["test_count"],
            config.metric_weights,
        ),
        index=frame.index,
        name="weight",
    )
    user_weights = compute_static_user_weights(frame, config)
    return row_weights, user_weights


def compute_legacy_polynomial_row_weights(frame: pd.DataFrame, config: ScoringConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Approximate the historic per-user fifth-degree polynomial weighting quickly."""

    work = _ensure_user_index(frame)
    cumulative_weights, user_weights = compute_time_row_weights(work, config)
    output = pd.Series(0.0, index=work.index, name="weight")

    user_indices = work.groupby("user_key", sort=False, dropna=False).indices
    user_index_values = work["_user_index"].to_numpy(dtype="float64")
    cumulative_values = cumulative_weights.to_numpy(dtype="float64")

    for _, indices in user_indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        n_classifications = len(indices)
        x_all = user_index_values[indices]
        y_all = cumulative_values[indices]
        capped = x_all <= config.max_user_classifications

        if n_classifications >= 100:
            sample_mask = capped & ((x_all % 10) == 0)
            x_sample = x_all[sample_mask]
            y_sample = y_all[sample_mask]
        else:
            sample_mask = capped
            x_sample = x_all[sample_mask]
            y_sample = y_all[sample_mask]

        if len(x_sample) == 0:
            continue
        if len(x_sample) == 1:
            output.iloc[indices] = y_sample[0]
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            polynomial = Polynomial.fit(x_sample, y_sample, deg=5)

        eval_x = np.minimum(x_all, float(config.max_user_classifications))
        output.iloc[indices] = polynomial(eval_x)

    return output, user_weights


def row_weights_for_mode(frame: pd.DataFrame, config: ScoringConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Dispatch thumbnail scoring to the requested user-weighting mode."""

    mode = config.mode.lower()
    if mode == "naive":
        user_weights = compute_static_user_weights(frame, config)
        mapped = frame["user_key"].map(user_weights.set_index("user_key")["weight"]).fillna(0.0)
        return mapped.astype("float64").rename("weight"), user_weights
    if mode == "time":
        return compute_time_row_weights(frame, config)
    if mode == "legacy":
        return compute_legacy_polynomial_row_weights(frame, config)
    raise ValueError(f"Unknown scoring mode: {config.mode}")


def _first_non_null(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def summarize_thumbnails(frame: pd.DataFrame) -> pd.DataFrame:
    """Build Zooniverse Toolz-style summary columns from raw normalized rows.

    These counts intentionally run before score-time vote caps and user/thumb
    deduping so the output CSVs match ``thumbnail_summary.csv`` semantics.
    """

    columns = ["thumbname", *THUMBNAIL_SUMMARY_COLUMNS]
    if frame.empty or "thumbname" not in frame.columns:
        return pd.DataFrame(columns=columns)

    thumbs = frame.dropna(subset=["thumbname"]).copy()
    if thumbs.empty:
        return pd.DataFrame(columns=columns)

    if "train" not in thumbs.columns:
        thumbs["train"] = False
    if "class" not in thumbs.columns:
        thumbs["class"] = pd.NA

    thumbs["_yes"] = thumbs["answer"].eq(1.0).astype("int64")
    thumbs["_no"] = thumbs["answer"].eq(0.0).astype("int64")
    grouped = thumbs.groupby("thumbname", sort=False, dropna=False).agg(
        **{
            "number_of_times_classified": ("thumbname", "size"),
            "number_of_yes_answers": ("_yes", "sum"),
            "number_of_no_answers": ("_no", "sum"),
            "train": ("train", _first_non_null),
            "class": ("class", _first_non_null),
        }
    )
    total_answers = grouped["number_of_yes_answers"] + grouped["number_of_no_answers"]
    grouped["fraction_yes"] = np.where(
        total_answers > 0,
        grouped["number_of_yes_answers"] / total_answers,
        np.nan,
    )
    grouped = grouped.reset_index()
    return grouped[columns]


def score_classifications(frame: pd.DataFrame, config: ScoringConfig) -> ScoreResult:
    """Score each thumbnail in a normalized classification dataframe."""

    work = _ensure_user_index(frame)
    thumbnail_summary = summarize_thumbnails(work)
    row_weights, user_weights = row_weights_for_mode(work, config)
    user_counts = work.groupby("user_key", sort=False, dropna=False).size()

    valid = work["thumbname"].notna() & work["answer"].isin([0.0, 1.0]) & work["user_key"].notna()
    if config.min_user_classifications > 1:
        valid &= work["user_key"].map(user_counts) >= config.min_user_classifications

    votes = work.loc[valid, ["row_order", "thumbname", "user_key", "answer", "batch"]].copy()
    votes["weight"] = row_weights.loc[votes.index].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if config.max_votes_per_thumb is not None and config.max_votes_per_thumb > 0:
        votes = votes.sort_values(["thumbname", "row_order"], kind="mergesort")
        vote_number = votes.groupby("thumbname", sort=False).cumcount()
        votes = votes.loc[vote_number < config.max_votes_per_thumb]

    if config.dedupe_user_thumbnail:
        votes = votes.drop_duplicates(["thumbname", "user_key"], keep="last")

    votes["weighted_yes"] = votes["weight"] * votes["answer"]
    grouped = votes.groupby("thumbname", sort=False, dropna=False).agg(
        numerator=("weighted_yes", "sum"),
        denominator=("weight", "sum"),
        vote_count=("answer", "size"),
        yes_count=("answer", "sum"),
        batch=("batch", _first_non_null),
    )
    denominator = grouped["denominator"].to_numpy(dtype="float64")
    score = np.zeros(len(grouped), dtype="float64")
    np.divide(
        grouped["numerator"].to_numpy(dtype="float64"),
        denominator,
        out=score,
        where=denominator != 0,
    )
    grouped["score"] = score

    if config.score_all_same_shortcut:
        grouped.loc[grouped["yes_count"] == grouped["vote_count"], "score"] = 1.0
        grouped.loc[grouped["yes_count"] == 0, "score"] = 0.0

    all_thumbs = (
        work.dropna(subset=["thumbname"])
        .groupby("thumbname", sort=False, dropna=False)
        .agg(batch=("batch", _first_non_null))
        .reset_index()
    )
    scores = all_thumbs.merge(grouped[["score", "vote_count", "batch"]], on="thumbname", how="left", suffixes=("", "_scored"))
    scores["score"] = scores["score"].fillna(0.0).astype("float64")
    scores["vote_count"] = scores["vote_count"].fillna(0).astype("int64")
    scores["batch"] = scores["batch_scored"].combine_first(scores["batch"])
    scores = scores.merge(thumbnail_summary, on="thumbname", how="left")
    scores = scores[["thumbname", "score", "vote_count", "batch", *THUMBNAIL_SUMMARY_COLUMNS]]

    diagnostics = {
        "mode": config.mode,
        "rows": int(len(work)),
        "thumbnails": int(scores["thumbname"].nunique()),
        "scored_votes": int(len(votes)),
        "users": int(work["user_key"].nunique(dropna=True)),
        "score_min": float(scores["score"].min()) if not scores.empty else math.nan,
        "score_max": float(scores["score"].max()) if not scores.empty else math.nan,
        "score_mean": float(scores["score"].mean()) if not scores.empty else math.nan,
        "above_cutoff": int((scores["score"] >= config.cutoff).sum()),
    }
    return ScoreResult(scores=scores, user_weights=user_weights, diagnostics=diagnostics)


def default_label(today: Optional[date] = None) -> str:
    day = today or date.today()
    return f"{day.year}_{day.month}_{day.day}"


def enrich_scores_with_coc(
    scores: pd.DataFrame,
    coc_scores_path: str | Path | None = DEFAULT_COC_SCORES_PATH,
    *,
    require_coc_scores: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Left-join COC sidecar scores onto generated thumbnail scores."""

    output = scores.copy()
    if coc_scores_path is None:
        output["coc_score"] = pd.NA
        return output, _empty_coc_diagnostics("disabled", "", len(output))

    coc_path = Path(coc_scores_path)
    if not coc_path.is_file():
        if require_coc_scores:
            raise FileNotFoundError(f"COC score file not found: {coc_path}")
        output["coc_score"] = pd.NA
        return output, _empty_coc_diagnostics("missing_default", str(coc_path), len(output))

    coc_scores, diagnostics = load_coc_scores(coc_path)
    output = output.drop(columns=["coc_score"], errors="ignore").merge(coc_scores, on="thumbname", how="left")
    matched = int(output["coc_score"].notna().sum()) if "coc_score" in output.columns else 0
    diagnostics.update(
        {
            "coc_scores_status": "loaded",
            "coc_scores_path": str(coc_path),
            "coc_matched_rows": matched,
            "coc_missing_rows": int(len(output) - matched),
        }
    )
    return output, diagnostics


def load_coc_scores(path: str | Path) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Load COC scores keyed by normalized thumbnail basename."""

    input_path = Path(path)
    frame = pd.read_csv(input_path, low_memory=False)
    score_column = _find_input_column(frame.columns, COC_SCORE_COLUMNS)
    if score_column is None:
        raise ValueError(f"No score column found in COC score file: {input_path}")

    key_source = coalesce_present_columns(frame, COC_THUMBNAIL_COLUMNS)
    path_source = coalesce_present_columns(frame, COC_PATH_COLUMNS)
    key_source = key_source.combine_first(path_source)

    work = pd.DataFrame(
        {
            "thumbname": normalize_thumbnames(key_source),
            "coc_score": pd.to_numeric(frame[score_column], errors="coerce"),
            "_coc_order": np.arange(len(frame), dtype="int64"),
        }
    )
    parsed = work.dropna(subset=["thumbname", "coc_score"]).copy()
    parsed = parsed.loc[parsed["thumbname"].astype("string").str.len() > 0]
    deduped = parsed.sort_values("_coc_order", kind="mergesort").drop_duplicates("thumbname", keep="last")

    diagnostics: dict[str, int | str] = {
        "coc_input_rows": int(len(frame)),
        "coc_parsed_rows": int(len(parsed)),
        "coc_skipped_rows": int(len(frame) - len(parsed)),
        "coc_unique_thumbnails": int(deduped["thumbname"].nunique()) if not deduped.empty else 0,
        "coc_duplicate_rows_removed": int(len(parsed) - len(deduped)),
    }
    return deduped[["thumbname", "coc_score"]].reset_index(drop=True), diagnostics


def _empty_coc_diagnostics(status: str, path: str, missing_rows: int) -> dict[str, int | str]:
    return {
        "coc_scores_status": status,
        "coc_scores_path": path,
        "coc_input_rows": 0,
        "coc_parsed_rows": 0,
        "coc_skipped_rows": 0,
        "coc_unique_thumbnails": 0,
        "coc_duplicate_rows_removed": 0,
        "coc_matched_rows": 0,
        "coc_missing_rows": missing_rows,
    }


def _find_input_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {_column_key(column): column for column in columns}
    for candidate in candidates:
        column = normalized.get(_column_key(candidate))
        if column is not None:
            return column
    return None


def _column_key(value: object) -> str:
    return "".join(str(value).strip().lower().split()).replace("_", "")


def write_score_outputs(
    result: ScoreResult,
    output_dir: str | Path,
    label: Optional[str],
    cutoff: float,
    coc_scores_path: str | Path | None = DEFAULT_COC_SCORES_PATH,
    *,
    require_coc_scores: bool = False,
) -> Mapping[str, Path]:
    """Write full, cutoff-filtered, and volunteer-weight CSV outputs."""

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_label = label or default_label()
    total_path = outdir / f"output_total_{output_label}.csv"
    filtered_path = outdir / f"output_{output_label}.csv"
    weights_path = outdir / f"user_weights_{output_label}.csv"

    output, coc_diagnostics = enrich_scores_with_coc(
        result.scores,
        coc_scores_path,
        require_coc_scores=require_coc_scores,
    )
    if isinstance(result.diagnostics, MutableMapping):
        result.diagnostics.update(coc_diagnostics)

    output = output.reindex(columns=SCORE_OUTPUT_COLUMNS).copy()
    output.to_csv(total_path, index=False)
    output.loc[output["score"] >= cutoff].to_csv(filtered_path, index=False)
    result.user_weights.to_csv(weights_path, index=False)
    return {"total": total_path, "filtered": filtered_path, "user_weights": weights_path}


def _read_score_file(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path, low_memory=False)
    key_column = "thumbname" if "thumbname" in frame.columns else "thumb_name"
    if key_column not in frame.columns or "score" not in frame.columns:
        raise ValueError(f"Score file must have thumbname/thumb_name and score columns: {path}")
    work = frame[[key_column, "score"]].rename(columns={key_column: "thumbname"}).copy()
    work["thumbname"] = normalize_thumbnames(work["thumbname"])
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    duplicate_count = int(work["thumbname"].duplicated(keep=False).sum())
    work = work.dropna(subset=["thumbname"]).groupby("thumbname", sort=False, as_index=False).agg(score=("score", "mean"))
    return work, {"rows": int(len(frame)), "unique_thumbnails": int(work["thumbname"].nunique()), "duplicate_rows": duplicate_count}


def compare_score_files(reference_path: str | Path, candidate_path: str | Path, diff_output: str | Path | None = None) -> Mapping[str, float | int]:
    """Compare two total score files after normalizing duplicate thumbnail rows."""

    reference, reference_info = _read_score_file(reference_path)
    candidate, candidate_info = _read_score_file(candidate_path)
    joined = reference.merge(candidate, on="thumbname", how="outer", suffixes=("_reference", "_candidate"), indicator=True)
    matched = joined["_merge"] == "both"
    diff = joined.loc[matched, "score_candidate"] - joined.loc[matched, "score_reference"]

    report: dict[str, float | int] = {
        "reference_rows": reference_info["rows"],
        "candidate_rows": candidate_info["rows"],
        "reference_unique_thumbnails": reference_info["unique_thumbnails"],
        "candidate_unique_thumbnails": candidate_info["unique_thumbnails"],
        "reference_duplicate_rows": reference_info["duplicate_rows"],
        "candidate_duplicate_rows": candidate_info["duplicate_rows"],
        "matched_thumbnails": int(matched.sum()),
        "reference_only": int((joined["_merge"] == "left_only").sum()),
        "candidate_only": int((joined["_merge"] == "right_only").sum()),
    }
    if matched.any():
        report.update(
            {
                "pearson_r": float(joined.loc[matched, ["score_reference", "score_candidate"]].corr().iloc[0, 1]),
                "mean_difference": float(diff.mean()),
                "mean_absolute_difference": float(diff.abs().mean()),
                "median_absolute_difference": float(diff.abs().median()),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "max_absolute_difference": float(diff.abs().max()),
            }
        )
    if diff_output is not None:
        joined["difference"] = joined["score_candidate"] - joined["score_reference"]
        joined["absolute_difference"] = joined["difference"].abs()
        joined.sort_values("absolute_difference", ascending=False).to_csv(diff_output, index=False)
    return report


def inspect_inputs(old_csv: str | Path, new_parquet: str | Path) -> Mapping[str, object]:
    """Report how an old reparsed CSV maps onto a cleaned Parquet export."""

    old = load_classifications(old_csv)
    new = load_classifications(new_parquet)

    old_columns = list(pd.read_csv(old_csv, nrows=0).columns)
    try:
        import pyarrow.parquet as pq

        new_columns = pq.ParquetFile(new_parquet).schema_arrow.names
    except ImportError:
        new_columns = list(read_table(new_parquet).columns)

    old_valid = old["answer"].isin([0.0, 1.0])
    new_valid = new["answer"].isin([0.0, 1.0])
    old_thumbs = set(old["thumbname"].dropna().astype(str))
    new_thumbs = set(new["thumbname"].dropna().astype(str))

    return {
        "old_columns": old_columns,
        "new_columns": list(new_columns),
        "old_rows": int(len(old)),
        "new_rows": int(len(new)),
        "old_valid_answers": int(old_valid.sum()),
        "new_valid_answers": int(new_valid.sum()),
        "old_unique_thumbnails": int(len(old_thumbs)),
        "new_unique_thumbnails": int(len(new_thumbs)),
        "old_unique_users": int(old["user_key"].nunique(dropna=True)),
        "new_unique_users": int(new["user_key"].nunique(dropna=True)),
        "old_training_rows": int(old["train"].sum()),
        "new_training_rows": int(new["train"].sum()),
        "thumbs_only_in_old": int(len(old_thumbs - new_thumbs)),
        "thumbs_only_in_new": int(len(new_thumbs - old_thumbs)),
        "sample_thumbs_only_in_old": sorted(old_thumbs - new_thumbs)[:10],
        "sample_thumbs_only_in_new": sorted(new_thumbs - old_thumbs)[:10],
    }
