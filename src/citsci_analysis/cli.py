from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

from .consolidate import consolidate_score_directory
from .scoring import (
    DEFAULT_CUTOFF,
    DEFAULT_COC_SCORES_PATH,
    ScoringConfig,
    compare_score_files,
    inspect_inputs,
    load_classifications,
    load_problematic_training_thumbs,
    score_classifications,
    write_score_outputs,
)
from .rubin import RubinProgressEvent, RubinScoringConfig, score_rubin_file, write_rubin_outputs


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _progress_enabled(setting: str, stream: TextIO) -> bool:
    """Resolve the CLI progress setting against the current stderr stream."""

    if setting == "always":
        return True
    if setting == "never":
        return False
    return stream.isatty()


class _RubinProgressPrinter:
    """Render Rubin progress events without contaminating stdout JSON."""

    _MIN_INTERVAL_SECONDS = 0.25
    _BAR_WIDTH = 28

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.is_tty = stream.isatty()
        self._bar: Any = None
        self._phase_started = time.perf_counter()
        self._last_rendered = 0.0
        self._line_length = 0
        self._active_line = False

    def __call__(self, event: RubinProgressEvent) -> None:
        if event.event == "message":
            self.message(event.phase)
            return
        if event.event == "start":
            self._phase_started = time.perf_counter()
            self._last_rendered = 0.0
            self._start_bar(event)
            return
        if event.event == "update":
            self._render(event)
            return
        if event.event == "finish":
            self._render(event, force=True, finished=True)

    def message(self, text: str) -> None:
        if self._bar is not None:
            self._bar.write(text)
            return
        self._end_active_line()
        print(text, file=self.stream, flush=True)

    def _start_bar(self, event: RubinProgressEvent) -> None:
        self._close_bar()
        self._end_active_line()

        try:
            from tqdm import tqdm
        except ImportError:
            self._active_line = True
            self._render(event, force=True)
            return

        self._bar = tqdm(
            total=event.total_rows,
            desc=event.phase,
            unit=" rows",
            file=self.stream,
            dynamic_ncols=True,
        )
        if event.chunks:
            self._bar.set_postfix(chunks=f"{event.chunks:,}", refresh=False)

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _end_active_line(self) -> None:
        if self.is_tty and self._active_line:
            self.stream.write("\n")
            self.stream.flush()
        self._active_line = False
        self._line_length = 0

    def _render(self, event: RubinProgressEvent, *, force: bool = False, finished: bool = False) -> None:
        if self._bar is not None:
            delta = max(0, event.rows - int(self._bar.n))
            if delta:
                self._bar.update(delta)
            self._bar.set_postfix(chunks=f"{event.chunks:,}", refresh=force)
            if finished:
                self._close_bar()
            return

        now = time.perf_counter()
        if not force and now - self._last_rendered < self._MIN_INTERVAL_SECONDS:
            return

        elapsed = max(now - self._phase_started, 0.001)
        text = self._format(event, elapsed, finished)
        if self.is_tty:
            padding = " " * max(0, self._line_length - len(text))
            self.stream.write(f"\r{text}{padding}")
            if finished:
                self.stream.write("\n")
                self._active_line = False
                self._line_length = 0
            else:
                self._line_length = len(text)
            self.stream.flush()
        else:
            print(text, file=self.stream, flush=True)
            if finished:
                self._active_line = False

        self._last_rendered = now

    def _format(self, event: RubinProgressEvent, elapsed: float, finished: bool) -> str:
        if event.total_rows and event.total_rows > 0:
            fraction = min(max(event.rows / event.total_rows, 0.0), 1.0)
            complete = int(self._BAR_WIDTH * fraction)
            bar = "#" * complete + "-" * (self._BAR_WIDTH - complete)
            text = (
                f"{event.phase}: [{bar}] {event.rows:,}/{event.total_rows:,} rows "
                f"({fraction * 100:5.1f}%), {event.chunks:,} chunks"
            )
        else:
            text = f"{event.phase}: {event.rows:,} rows, {event.chunks:,} chunks"

        if event.rows > 0:
            text += f", {event.rows / elapsed:,.0f} rows/s"
        if finished:
            text += f", done in {elapsed:.1f}s"
        return text


def _progress_message(enabled: bool, stream: TextIO, text: str) -> None:
    if enabled:
        print(text, file=stream, flush=True)


def add_workset_options(parser: argparse.ArgumentParser, *, standalone: bool = False) -> None:
    parser.add_argument("--workset-root", type=Path, help="Shared candidate workset destination; defaults to mounted Dropbox when available.")
    parser.add_argument("--source-root", type=Path, action="append", default=None, help="Repeatable image source folder (RCC sources are read only).")
    parser.add_argument("--source-index", type=Path, action="append", default=None, help="Repeatable AA archive CSV/Parquet index.")
    parser.add_argument("--data-root", type=Path, default=None, help="Mounted data root for relative archive paths.")
    parser.add_argument("--training-metadata", type=Path, help="Thumbnail summary or normalized classifications with explicit training flags.")
    parser.add_argument("--training-null-is-nontraining", action="store_true", help="Explicitly apply the Zooniverse nullable training convention; unknown nonempty values remain unresolved.")
    parser.add_argument("--context", type=int, default=None, help="AA cutouts on each side of every qualifying image (default: 3).")
    parser.add_argument("--remote", help="AA SSH archive host; defaults to the configured Arnor source.")
    parser.add_argument("--remote-path-map", action="append", default=[], metavar="OLD=NEW", help="Repeatable explicit remapping of stale remote index path prefixes.")
    parser.add_argument("--remote-index", help="AA remote PNG index path.")
    parser.add_argument("--no-remote", action="store_true", help="Use local archive inventories only.")
    parser.add_argument("--migrate-legacy", action="store_true", help="Journal and move verified legacy root PNGs into object folders.")
    parser.add_argument("--inventory-timeout", type=int, default=600, help="Maximum seconds for the filtered AA remote index query.")
    parser.add_argument("--transfer-timeout", type=int, default=600, help="Maximum total seconds across resumable transfer batches.")
    parser.add_argument("--transfer-batch-size", type=int, default=512, help="Maximum AA paths per transfer batch (default 512).")
    if standalone:
        parser.add_argument("--preview", action="store_true", help="Read-only selection and inventory preview; no image transfers or output writes.")
    else:
        parser.add_argument("--no-workset", action="store_true", help="Disable the default-on candidate workset stage.")


def run_workset(args: argparse.Namespace, scores: Path, project: str, training: Path | None = None) -> dict:
    from .worksets import WorksetConfig, default_destination, default_sources, prepare_workset
    destination = args.workset_root or default_destination()
    if destination is None:
        destination = getattr(args, "output_dir", scores.parent) / "candidate_worksets"
    roots, indexes = default_sources(destination)
    if project == "rcc":
        roots, indexes = (), ()
    options = dict(project=project, scores=scores, destination=destination,
                   training=args.training_metadata or training,
                   source_roots=tuple(args.source_root) if args.source_root is not None else roots,
                   source_indexes=tuple(args.source_index) if args.source_index is not None else indexes,
                   cutoff=args.cutoff, context=args.context, label=args.label or "workset",
                   preview=getattr(args, "preview", False), migrate=args.migrate_legacy,
                   progress=_progress_enabled(getattr(args, "progress", "always"), sys.stderr),
                   inventory_timeout=args.inventory_timeout, transfer_timeout=args.transfer_timeout, transfer_batch_size=args.transfer_batch_size, training_null_is_nontraining=args.training_null_is_nontraining)
    if args.data_root is not None:
        options["data_root"] = args.data_root
    if args.no_remote:
        options["remote"] = None
    elif args.remote:
        options["remote"] = args.remote
    if args.remote_index:
        options["remote_index"] = args.remote_index
    mappings = []
    for value in args.remote_path_map:
        if "=" not in value:
            raise ValueError("Remote path mapping requires OLD=NEW")
        mappings.append(tuple(value.split("=", 1)))
    options["remote_path_maps"] = tuple(mappings)
    return prepare_workset(WorksetConfig(**options))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score citizen-science Zooniverse classifications.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score thumbnails/review units from CSV or cleaned Parquet classifications.")
    score.add_argument("input", type=Path, help="Path to reparsed.csv(.gz) or cleaned_classifications.parquet.")
    score.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for output CSVs.")
    score.add_argument("--label", default=None, help="Filename date/label, e.g. 2025_1_13. Defaults to today.")
    score.add_argument("--mode", choices=["naive", "time", "legacy", "rubin"], default="naive")
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
        help="Count repeated classifications by the same user on the same thumbnail/review unit.",
    )
    score.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for Rubin CSV/Parquet scoring.",
    )
    score.add_argument(
        "--progress",
        choices=["auto", "always", "never"],
        default="auto",
        help="Show scoring progress on stderr. Defaults to auto in interactive terminals.",
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
    coc_group = score.add_mutually_exclusive_group()
    coc_group.add_argument(
        "--coc-scores",
        type=Path,
        default=None,
        help=f"COC score CSV to join as coc_score. Defaults to {DEFAULT_COC_SCORES_PATH} when present.",
    )
    coc_group.add_argument(
        "--no-coc-scores",
        action="store_true",
        help="Disable COC score enrichment and leave coc_score blank.",
    )

    add_workset_options(score)
    workset = subparsers.add_parser("prepare-workset", help="Prepare or preview candidates from an existing full score CSV.")
    workset.add_argument("scores", type=Path)
    workset.add_argument("--project", required=True, choices=["aa", "rcc"])
    workset.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    workset.add_argument("--label", default="workset")
    add_workset_options(workset, standalone=True)

    compare = subparsers.add_parser("compare", help="Compare two output_total score files.")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--diff-output", type=Path, default=None, help="Optional CSV with per-thumbnail differences.")

    inspect = subparsers.add_parser("inspect-inputs", help="Compare the old reparsed CSV and new cleaned Parquet inputs.")
    inspect.add_argument("old_csv", type=Path)
    inspect.add_argument("new_parquet", type=Path)

    consolidate = subparsers.add_parser(
        "consolidate-scores",
        help="Consolidate recursive score CSVs to the latest score per thumbnail basename.",
    )
    consolidate.add_argument("directory", type=Path, help="Directory to search recursively for CSV files.")
    consolidate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to consolidated_scores.csv in the input directory.",
    )
    consolidate.add_argument(
        "--progress",
        choices=["auto", "always", "never"],
        default="auto",
        help="Show consolidation progress on stderr. Defaults to auto in interactive terminals.",
    )

    return parser


def run_consolidate_scores(args: argparse.Namespace) -> int:
    progress_stream = sys.stderr
    show_progress = _progress_enabled(args.progress, progress_stream)
    progress = (lambda text: _progress_message(True, progress_stream, text)) if show_progress else None

    result = consolidate_score_directory(args.directory, args.output, progress)
    _print_json(result.summary)
    return 0


def run_score(args: argparse.Namespace) -> int:
    progress_stream = sys.stderr
    show_progress = _progress_enabled(args.progress, progress_stream)

    problematic = frozenset()
    if args.problematic_training is not None:
        _progress_message(show_progress, progress_stream, "Loading problematic training list...")
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
    if args.mode == "rubin":
        config = RubinScoringConfig(
            metric_weights=tuple(args.weights),
            max_user_classifications=args.max_user_classifications,
            min_user_classifications=min_user_classifications,
            cutoff=args.cutoff,
            chunk_size=args.chunk_size,
            dedupe_user_review_units=not args.keep_duplicate_user_thumbs,
            problematic_training_review_units=problematic,
        )
        progress = _RubinProgressPrinter(progress_stream) if show_progress else None
        loaded = time.perf_counter()
        result = score_rubin_file(args.input, config, progress)
        scored = time.perf_counter()
        if progress is not None:
            progress.message("Writing Rubin outputs...")
        paths = write_rubin_outputs(result, args.output_dir, args.label, args.cutoff)
        written = time.perf_counter()
        result.diagnostics.update(
            {
                "load_seconds": round(loaded - start, 3),
                "score_seconds": round(scored - loaded, 3),
                "write_seconds": round(written - scored, 3),
                "total_seconds": round(written - start, 3),
            }
        )
        workset = None if args.no_workset else run_workset(args, paths["review_units"], "rcc", args.input)
        result.diagnostics.update(workset_seconds=round(time.perf_counter() - written, 3), total_seconds=round(time.perf_counter() - start, 3))
        _print_json({"diagnostics": result.diagnostics, "outputs": paths, "workset": workset})
        return 0

    _progress_message(show_progress, progress_stream, "Loading input table...")
    frame = load_classifications(args.input)
    loaded = time.perf_counter()
    _progress_message(show_progress, progress_stream, "Scoring classifications...")
    result = score_classifications(frame, config)
    scored = time.perf_counter()
    _progress_message(show_progress, progress_stream, "Writing score outputs...")
    coc_scores_path = None if args.no_coc_scores else args.coc_scores or DEFAULT_COC_SCORES_PATH
    paths = write_score_outputs(
        result,
        args.output_dir,
        args.label,
        args.cutoff,
        coc_scores_path=coc_scores_path,
        require_coc_scores=args.coc_scores is not None,
    )
    written = time.perf_counter()
    result.diagnostics.update(
        {
            "load_seconds": round(loaded - start, 3),
            "score_seconds": round(scored - loaded, 3),
            "write_seconds": round(written - scored, 3),
            "total_seconds": round(written - start, 3),
        }
    )
    workset = None if args.no_workset else run_workset(args, paths["total"], "aa", args.input)
    result.diagnostics.update(workset_seconds=round(time.perf_counter() - written, 3), total_seconds=round(time.perf_counter() - start, 3))
    _print_json({"diagnostics": result.diagnostics, "outputs": paths, "workset": workset})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-workset":
        training = args.scores.parent.parent / "reparsed_z/cleaned_classifications.parquet"
        if args.project == "aa" or not training.is_file():
            training = args.scores.parent.parent / "reparsed_z/thumbnail_summary.csv"
        _print_json(run_workset(args, args.scores, args.project, training if training.is_file() else None))
        return 0
    if args.command == "score":
        return run_score(args)
    if args.command == "compare":
        _print_json(compare_score_files(args.reference, args.candidate, args.diff_output))
        return 0
    if args.command == "inspect-inputs":
        _print_json(inspect_inputs(args.old_csv, args.new_parquet))
        return 0
    if args.command == "consolidate-scores":
        return run_consolidate_scores(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
