from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .scoring import (
    DEFAULT_CUTOFF,
    ScoringConfig,
    compare_score_files,
    inspect_inputs,
    load_classifications,
    load_problematic_training_thumbs,
    score_classifications,
    write_score_outputs,
)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Active Asteroids Zooniverse classifications.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score thumbnails from old CSV or cleaned Parquet classifications.")
    score.add_argument("input", type=Path, help="Path to reparsed.csv(.gz) or cleaned_classifications.parquet.")
    score.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for output CSVs.")
    score.add_argument("--label", default=None, help="Filename date/label, e.g. 2025_1_13. Defaults to today.")
    score.add_argument("--mode", choices=["naive", "time", "legacy"], default="naive")
    score.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    score.add_argument("--weights", nargs=3, type=float, metavar=("A", "B", "C"), default=(7.0, 2.0, 1.0))
    score.add_argument("--max-user-classifications", type=int, default=10_000)
    score.add_argument(
        "--min-user-classifications",
        type=int,
        default=None,
        help="Minimum total classifications for a volunteer vote to count. Defaults to 4 for legacy, 1 otherwise.",
    )
    score.add_argument(
        "--max-votes-per-thumb",
        type=int,
        default=None,
        help="Use only the first N votes per thumbnail. Defaults to 15 for legacy, all votes otherwise.",
    )
    score.add_argument(
        "--keep-duplicate-user-thumbs",
        action="store_true",
        help="Count repeated classifications by the same user on the same thumbnail.",
    )
    score.add_argument(
        "--all-same-shortcut",
        action="store_true",
        help="Force score 1/0 when all counted votes for a thumbnail are yes/no.",
    )
    score.add_argument(
        "--problematic-training",
        type=Path,
        default=None,
        help="Optional pickle/CSV with thumb_name and Problematic Yes/No columns.",
    )

    compare = subparsers.add_parser("compare", help="Compare two output_total score files.")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--diff-output", type=Path, default=None, help="Optional CSV with per-thumbnail differences.")

    inspect = subparsers.add_parser("inspect-inputs", help="Compare the old reparsed CSV and new cleaned Parquet inputs.")
    inspect.add_argument("old_csv", type=Path)
    inspect.add_argument("new_parquet", type=Path)

    return parser


def run_score(args: argparse.Namespace) -> int:
    problematic = frozenset()
    if args.problematic_training is not None:
        problematic = load_problematic_training_thumbs(args.problematic_training)

    min_user_classifications = args.min_user_classifications
    if min_user_classifications is None:
        min_user_classifications = 4 if args.mode == "legacy" else 1

    max_votes_per_thumb = args.max_votes_per_thumb
    if max_votes_per_thumb is None and args.mode == "legacy":
        max_votes_per_thumb = 15

    config = ScoringConfig(
        mode=args.mode,
        metric_weights=tuple(args.weights),
        max_user_classifications=args.max_user_classifications,
        min_user_classifications=min_user_classifications,
        max_votes_per_thumb=max_votes_per_thumb,
        dedupe_user_thumbnail=not args.keep_duplicate_user_thumbs,
        score_all_same_shortcut=args.all_same_shortcut or args.mode == "legacy",
        cutoff=args.cutoff,
        problematic_training_thumbs=problematic,
    )

    start = time.perf_counter()
    frame = load_classifications(args.input)
    loaded = time.perf_counter()
    result = score_classifications(frame, config)
    scored = time.perf_counter()
    paths = write_score_outputs(result, args.output_dir, args.label, args.cutoff)
    written = time.perf_counter()
    result.diagnostics.update(
        {
            "load_seconds": round(loaded - start, 3),
            "score_seconds": round(scored - loaded, 3),
            "write_seconds": round(written - scored, 3),
            "total_seconds": round(written - start, 3),
        }
    )
    _print_json({"diagnostics": result.diagnostics, "outputs": paths})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "score":
        return run_score(args)
    if args.command == "compare":
        _print_json(compare_score_files(args.reference, args.candidate, args.diff_output))
        return 0
    if args.command == "inspect-inputs":
        _print_json(inspect_inputs(args.old_csv, args.new_parquet))
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
