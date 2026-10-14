"""Resumable candidate worksets. Score computation is deliberately independent.

Contract v1 uses project/object/logical-PNG-name identities; presentation paths
and run labels never participate in classification identity.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NotRequired, TypeAlias, TypedDict, cast

import pandas as pd

# Heterogeneous manifest/CSV records are an intentional serialization boundary.
Record: TypeAlias = dict[str, Any]


class Source(TypedDict):
    """An indexed image location, local or on the configured SSH host."""

    path: str
    remote: NotRequired[bool]


SourceMap: TypeAlias = Mapping[str, list[Source]]
Inventory: TypeAlias = defaultdict[str, defaultdict[str, list[Source]]]
ScoreLookup: TypeAlias = dict[str, tuple[float, str]]


DROPBOX_SUFFIX = Path("School/Research/Citizen Science Stuff/Active Asteroids Zooniverse Citizen Science")
DEFAULT_REMOTE = "coc123@arnor.astro.washington.edu"
DEFAULT_REMOTE_INDEX = "/astro/store/epyc3/projects3/active-asteroids/SAFARI2/Data/D0807/all_png_epyc.lst"
DEFAULT_REMOTE_PATH_MAPS = (
    (
        "/epyc/projects3/active-asteroids/SAFARI2/Data",
        "/astro/store/epyc3/projects3/active-asteroids/SAFARI2/Data",
    ),
)
DEFAULT_DATA_ROOT = Path("/Volumes/uranus/monsoon/projects/asteroids/SAFARI2/Data")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
AA_DEDUPLICATION = "object-exposure-datetime-pI-v1"
OBJECT_FIELDS = [
    "project",
    "object_id",
    "object_class",
    "object_cs_score",
    "selection_method",
    "qualifying_count",
    "best_image",
    "image_count",
    "available_count",
]
IMAGE_FIELDS = [
    "image_id",
    "object_id",
    "review_object_name",
    "object_class",
    "product",
    "filename",
    "order",
    "membership",
    "relative_path",
    "source_path",
    "availability",
    "sha256",
]
REVEAL_FIELDS = ["image_id", "object_id", "cs_score", "coc_score", "object_cs_score"]


@dataclass
class WorksetConfig:
    """Selection, source inventory, and publication settings for a workset."""

    project: str
    scores: Path
    destination: Path
    training: Path | None = None
    source_roots: tuple[Path, ...] = ()
    source_indexes: tuple[Path, ...] = ()
    data_root: Path = DEFAULT_DATA_ROOT
    cutoff: float = 0.473
    context: int | None = None
    label: str = "workset"
    remote: str | None = DEFAULT_REMOTE
    remote_index: str = DEFAULT_REMOTE_INDEX
    remote_path_maps: tuple[tuple[str, str], ...] = ()
    progress: bool = True
    preview: bool = False
    migrate: bool = False
    inventory_timeout: int = 600
    transfer_timeout: int = 600
    transfer_batch_size: int = 512
    training_null_is_nontraining: bool = False

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = 3 if self.project == "aa" else 10
        self.scores = Path(self.scores).expanduser()
        self.destination = Path(self.destination).expanduser()
        self.training = Path(self.training).expanduser() if self.training else None
        self.data_root = Path(self.data_root).expanduser()
        self.source_roots = tuple(Path(p).expanduser() for p in self.source_roots)
        self.source_indexes = tuple(Path(p).expanduser() for p in self.source_indexes)
        if self.project == "aa" and self.remote == DEFAULT_REMOTE and not self.remote_path_maps:
            self.remote_path_maps = DEFAULT_REMOTE_PATH_MAPS
        if self.inventory_timeout <= 0 or self.transfer_timeout <= 0 or self.transfer_batch_size <= 0:
            raise ValueError("Timeouts and transfer batch size must be positive")
        for old, new in self.remote_path_maps:
            if not all(
                PurePosixPath(v).is_absolute()
                and ".." not in PurePosixPath(v).parts
                and not any(c in v for c in "\n\r\0")
                for v in (old, new)
            ):
                raise ValueError("Remote path maps must use absolute safe prefixes")
        if self.remote and (self.remote.startswith("-") or any(c.isspace() for c in self.remote)):
            raise ValueError("Invalid SSH host")


def emit_progress(config: WorksetConfig, message: str) -> None:
    """Write an enabled workset progress message to standard error."""

    if config.progress:
        print(message, file=sys.stderr, flush=True)


def default_destination() -> Path | None:
    """Find the configured or mounted default candidate workset directory."""

    explicit = os.environ.get("CITSCI_WORKSET_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    for base in [Path.home() / "Dropbox", Path.home() / "Colinchandler Dropbox/Colin Chandler"]:
        candidate = base / DROPBOX_SUFFIX / "all_interesting"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def default_sources(destination: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Find existing local candidate roots and archive index files."""

    base = destination.parent
    roots = tuple(
        p for p in (base / "Candidates/all_via_analysis", base / "Candidates/followup") if p.is_dir()
    )
    manifests = DEFAULT_DATA_ROOT.parent / "Library/Manifests"
    indexes = tuple(
        p
        for p in (
            manifests / "monsoon_projects_cutouts.parquet",
            manifests / "monsoon_scratch_cutouts.parquet",
        )
        if p.is_file()
    )
    return roots, indexes


def logical_name(value: object) -> str:
    """Normalize an image path to its logical PNG basename."""

    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".png.gz"):
        return name[:-3]
    for suffix in (".fits.fz", ".fits", ".png"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] + ".png"
    return name + ".png" if name else ""


def safe_component(value: str) -> str:
    """Validate a single path component before constructing output paths."""

    if not value or value in {".", ".."} or any(c in value for c in "/\\\n\r\0"):
        raise ValueError("Unsafe or empty identity component")
    return value


def image_key(project: str, obj: str, filename: str) -> str:
    """Build a stable project, object, and image identity."""

    return hashlib.sha256(json.dumps([project, obj, filename], separators=(",", ":")).encode()).hexdigest()


def digest(path: Path) -> str:
    """Compute the SHA-256 digest of a file without loading it all at once."""

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def training_value(value: object) -> bool | None:
    """Interpret an explicit training flag, returning None when unknown."""

    value = str(value).strip().lower()
    if value in {"1", "1.0", "true", "yes", "t"}:
        return True
    if value in {"0", "0.0", "false", "no", "f"}:
        return False
    return None


def read_training(path: Path, wanted: set[str], null_is_nontraining: bool = False) -> dict[str, set[bool]]:
    """Read training evidence for selected images from CSV or Parquet."""

    result: dict[str, set[bool]] = defaultdict(set)
    chunks: Iterable[pd.DataFrame]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(path)
        cols = [
            c
            for c in ["review_unit_id", "thumbname", "filename", "train", "training_subject"]
            if c in table.schema_arrow.names
        ]
        chunks = (b.to_pandas().fillna("") for b in table.iter_batches(columns=cols, batch_size=100_000))
    else:
        chunks = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000)
    for chunk in chunks:
        for row in chunk.to_dict("records"):
            key = logical_name(row.get("review_unit_id") or row.get("thumbname") or row.get("filename"))
            if key not in wanted:
                continue
            # The dedicated subject flag distinguishes negative training controls
            # (legacy Train=0) from non-training subjects.
            subject_flag = training_value(row.get("training_subject"))
            flag = subject_flag if subject_flag is not None else training_value(row.get("train"))
            if (
                flag is None
                and null_is_nontraining
                and "train" in row
                and not str(row.get("train") or "").strip()
            ):
                flag = False
            if flag is not None:
                result[key].add(flag)
    return result


def select_candidates(
    config: WorksetConfig,
) -> tuple[list[Record], dict[str, set[str]], ScoreLookup, list[Record]]:
    """Select non-training objects and votes, retaining score provenance and issues."""

    if (
        config.project not in {"aa", "rcc"}
        or not math.isfinite(config.cutoff)
        or not 0 <= config.cutoff <= 1
        or cast(int, config.context) < 0
    ):
        raise ValueError("Invalid project, cutoff, or context count")
    frame = pd.read_csv(config.scores, dtype=str, keep_default_na=False)
    keycol = "thumbname" if config.project == "aa" else "review_unit_id"
    required = {keycol, "score"} | ({"object_id"} if config.project == "rcc" else set())
    if not required <= set(frame):
        raise ValueError(f"Score CSV requires {sorted(required)}")
    frame["filename"] = frame[keycol].map(logical_name)
    if frame["filename"].duplicated().any():
        raise ValueError("Duplicate logical image identities in score CSV")
    flags = (
        read_training(config.training, set(frame.filename), config.training_null_is_nontraining)
        if config.training
        else {}
    )
    issues = []
    groups = defaultdict(list)
    score_lookup = {}
    for row in frame.to_dict("records"):
        name = row["filename"]
        try:
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError()
        except ValueError:
            issues.append({"image": name, "reason": "invalid_score"})
            continue
        values = set(flags.get(name, set()))
        flag = training_value(row.get("train"))
        if flag is not None:
            values.add(flag)
        coc = row.get("coc_score", "")
        if coc:
            try:
                if not math.isfinite(float(coc)):
                    raise ValueError()
            except ValueError:
                issues.append({"image": name, "reason": "invalid_coc_score"})
                coc = ""
        score_lookup[name] = (score, coc)
        if len(values) != 1:
            issues.append({"image": name, "reason": "conflicting_training" if values else "missing_training"})
            continue
        if True in values:
            continue
        obj = safe_component(name.split("_", 1)[0] if config.project == "aa" else row["object_id"])
        cls = row.get("class", "") if config.project == "aa" else row.get("object_class", "")
        groups[obj].append((name, score, cls))
    objects: list[Record] = []
    targets: dict[str, set[str]] = {}
    for obj, items in groups.items():
        ranked = sorted(items, key=lambda r: (-r[1], r[0]))
        qualifying = [r[0] for r in ranked if r[1] >= config.cutoff]
        aggregate = (
            ranked[0][1] if config.project == "aa" else sum(r[1] for r in ranked[:3]) / len(ranked[:3])
        )
        if aggregate < config.cutoff:
            continue
        classes = {r[2] for r in items if r[2]}
        if len(classes) > 1:
            issues.append({"object": obj, "reason": "conflicting_object_class"})
            continue
        cls = next(iter(classes), "")
        if config.project == "aa" and not cls:
            issues.append({"object": obj, "reason": "missing_object_class"})
            continue
        objects.append(
            dict(
                project=config.project,
                object_id=obj,
                object_class=cls,
                object_cs_score=aggregate,
                selection_method="peak_nontraining" if config.project == "aa" else "top3_mean_nontraining",
                qualifying_count=len(qualifying),
                best_image=ranked[0][0],
            )
        )
        targets[obj] = set(qualifying if config.project == "aa" else [r[0] for r in ranked])
    objects.sort(key=lambda r: (-r["object_cs_score"], r["object_id"]))
    return objects, targets, score_lookup, issues


def observation_key(name: str) -> tuple[str, str]:
    # AA stores YYYYMMDD-HHMM plus the tens-of-seconds digit.
    """Order thumbnails by their encoded observation timestamp and name."""

    match = re.search(r"_(\d{8}-\d{5})_", name)
    return (match.group(1) if match else name, name)


def observation_identity(name: str) -> tuple[str, str]:
    """Keep unparseable names distinct instead of guessing an observation time."""
    match = re.match(r"^([^_]+)_(\d{8}-\d{5})_", name)
    if match:
        try:
            stamp = datetime.strptime(match[2] + "0", "%Y%m%d-%H%M%S")
            return match[1], stamp.strftime("%Y%m%d-%H%M%S")
        except ValueError:
            pass
    return name.split("_", 1)[0], name


def exposure_identity(name: str, sources: list[Source]) -> tuple[str, str] | None:
    """Use the DECam exposure timestamp, independent of processing version.

    Historical thumbnails can encode start or midpoint times minutes apart.
    Only an unambiguous source exposure can override the filename timestamp.
    """
    stamps = set()
    for source in sources:
        match = re.search(r"/c4d_(\d{6})_(\d{6})_[^/]+/", source["path"])
        if match:
            try:
                stamp = datetime.strptime("20" + match[1] + match[2], "%Y%m%d%H%M%S")
                stamps.add(stamp.strftime("%Y%m%d-%H%M%S"))
            except ValueError:
                pass
    if len(stamps) == 1:
        return name.split("_", 1)[0], next(iter(stamps))
    return None


def observation_representatives(
    names: set[str], targets: set[str], sources: SourceMap | None = None
) -> dict[str, str]:
    """Map each AA variant to one deterministic representative per observation.

    A filename-only entry can join a known exposure through its timestamp when
    that timestamp identifies exactly one exposure. Distinct known exposures
    are never merged merely because their truncated filename timestamps match.
    """
    sources = sources or {}
    names = names | targets
    exposures = {n: exposure_identity(n, sources.get(n, [])) for n in names}
    by_time = defaultdict(set)
    for name, exposure in exposures.items():
        if exposure is not None:
            by_time[observation_identity(name)].add(exposure)
    groups = defaultdict(list)
    for name in names:
        timestamp = observation_identity(name)
        matches = by_time[timestamp]
        if exposures[name] is not None:
            key = ("exposure", exposures[name])
        elif len(matches) == 1:
            key = ("exposure", next(iter(matches)))
        else:
            key = ("filename", timestamp)
        groups[key].append(name)
    representatives: dict[str, str] = {}
    for variants in groups.values():

        def preference(name: str) -> tuple[int, bool, str]:
            tokens = Path(name).stem.split("_")
            return (0 if "pI" in tokens else 1 if "p2" in tokens else 2, name not in targets, name)

        preferred = min(variants, key=preference)
        representatives.update((name, preferred) for name in variants)
    return representatives


def context_window(
    names: set[str], targets: set[str], count: int, sources: SourceMap | None = None
) -> list[str]:
    """Select observation representatives within each target neighbor window."""

    representatives = observation_representatives(names, targets, sources)
    targets = {representatives[n] for n in targets}
    ordered = sorted(
        set(representatives.values()),
        key=lambda n: (exposure_identity(n, (sources or {}).get(n, [])) or observation_identity(n), n),
    )
    selected = set()
    for i, name in enumerate(ordered):
        if name in targets:
            selected.update(ordered[max(0, i - count) : i + count + 1])
    return [name for name in ordered if name in selected]


def error_detail(exc: Exception) -> str:
    """Normalize subprocess or filesystem exception details for JSON reports."""

    detail = getattr(exc, "stderr", "") or str(exc)
    return (detail.decode(errors="replace") if isinstance(detail, bytes) else str(detail))[:500]


def run_command(
    args: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: float = 120,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Bound the whole SSH/rsync process group, including inherited children."""
    with subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input_text, timeout=timeout)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        result.check_returncode()
        return result


def ssh(
    remote: str, command: str, data: str | None = None, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """Execute a bounded non-interactive command on the configured SSH host."""

    if remote.startswith("-") or any(c.isspace() for c in remote):
        raise ValueError("Invalid SSH host")
    return run_command(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", remote, command],
        input_text=data,
        timeout=timeout,
    )


def inventory(config: WorksetConfig, targets: dict[str, set[str]]) -> tuple[Inventory, list[Record]]:
    """Read indexes once; never recursively traverse the SAFARI data tree."""
    found: Inventory = defaultdict(lambda: defaultdict(list))
    issues = []
    names_to_object = {name: obj for obj, names in targets.items() for name in names}

    def add(raw: object, source: str | Path, remote: bool = False) -> None:
        if not str(raw).lower().endswith((".png", ".png.gz")):
            return
        name = logical_name(raw)
        obj = name.split("_", 1)[0] if config.project == "aa" else names_to_object.get(name)
        if obj is not None and obj in targets:
            item: Source = {"path": str(source), "remote": remote}
            if item not in found[obj][name]:
                found[obj][name].append(item)

    for root in config.source_roots:
        if not root.is_dir():
            issues.append({"source": str(root), "reason": "missing_source_root"})
            continue
        for directory, dirs, files in os.walk(
            root,
            onerror=lambda exc: issues.append(
                {"source": str(exc.filename), "reason": "unreadable_source_directory"}
            ),
        ):
            dirs[:] = sorted(
                d for d in dirs if not d.startswith(".") and d not in {"worksets", "state", "migration"}
            )
            for name in sorted(files):
                add(name, Path(directory) / name)
    if config.project == "aa":
        # Shared images and legacy root entries are valid sources too.
        for root in (config.destination / "aa/objects",):
            if root.is_dir():
                for obj in sorted(targets):
                    for p in sorted((root / obj).glob("*.png")):
                        add(p.name, p)
        if config.destination.is_dir():
            for p in sorted(config.destination.glob("*.png")):
                add(p.name, p)
        for index in config.source_indexes:
            if not index.is_file():
                issues.append({"source": str(index), "reason": "missing_source_index"})
                continue
            chunks: Iterable[list[Record]]
            if index.suffix == ".parquet":
                import pyarrow.parquet as pq

                f = pq.ParquetFile(index)
                cols = [
                    c for c in ["basename", "relative_path", "absolute_path"] if c in f.schema_arrow.names
                ]
                chunks = (b.to_pylist() for b in f.iter_batches(columns=cols, batch_size=100_000))
            else:

                def csv_chunks(csv_path: Path = index) -> Iterator[list[dict[str, str]]]:
                    with csv_path.open(newline="") as stream:
                        yield from ([row] for row in csv.DictReader(stream))

                chunks = csv_chunks()
            for rows in chunks:
                for row in rows:
                    relative = row.get("relative_path", "")
                    name = row.get("basename") or Path(relative).name
                    if not name.lower().endswith((".png", ".png.gz")) or name.split("_", 1)[0] not in targets:
                        continue
                    rel = PurePosixPath(relative)
                    if rel.is_absolute() or ".." in rel.parts:
                        issues.append({"source": relative, "reason": "unsafe_index_path"})
                        continue
                    add(name, config.data_root / relative)
        if config.remote and targets:
            try:
                patterns = "".join("/" + safe_component(obj) + "_\n" for obj in sorted(targets))
                command = (
                    f"LC_ALL=C grep -F -f - -- {shlex.quote(config.remote_index)}; "
                    's=$?; if [ "$s" -eq 1 ]; then exit 0; fi; exit "$s"'
                )
                lines = ssh(
                    config.remote, command, patterns, timeout=config.inventory_timeout
                ).stdout.splitlines()
                for line in lines:
                    remote_path = PurePosixPath(line.strip())
                    if remote_path.is_absolute() and ".." not in remote_path.parts:
                        mapped = str(remote_path)
                        for old, new in config.remote_path_maps:
                            if mapped.startswith(old.rstrip("/") + "/"):
                                mapped = new.rstrip("/") + mapped[len(old.rstrip("/")) :]
                                break
                        add(remote_path.name, mapped, remote=True)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                issues.append({"reason": "remote_inventory_failed", "detail": error_detail(exc)})
    return found, issues


def png_bytes(path: Path) -> bytes:
    """Read and validate compressed or uncompressed PNG contents."""

    with gzip.open(path, "rb") if path.name.lower().endswith(".gz") else path.open("rb") as stream:
        data = stream.read()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    pos, seen_header, seen_data, ended = 8, False, False, False
    payloads = []
    while pos + 12 <= len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + size]
        if len(body) != size or pos + 12 + size > len(data):
            raise ValueError("Truncated PNG")
        crc = struct.unpack(">I", data[pos + 8 + size : pos + 12 + size])[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise ValueError("PNG checksum mismatch")
        if not seen_header:
            if kind != b"IHDR" or size != 13 or not all(struct.unpack(">II", body[:8])):
                raise ValueError("Invalid PNG header")
            seen_header = True
        if kind == b"IDAT":
            seen_data = True
            payloads.append(body)
        if kind == b"IEND":
            ended = True
            break
        pos += 12 + size
    if not (seen_header and seen_data and ended):
        raise ValueError("Incomplete PNG")
    decoder = zlib.decompressobj()
    decoder.decompress(b"".join(payloads))
    if not decoder.eof:
        raise ValueError("Truncated PNG image data")
    return data


def atomic_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a metadata file after flushing its contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish_png(path: Path, data: bytes) -> None:
    """Publish PNG bytes exclusively, preserving any existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".copy-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive publication: never replaces.
    finally:
        os.unlink(temporary)


def write_json(path: Path, data: object) -> None:
    """Serialize metadata deterministically and publish it atomically."""

    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write the requested CSV fields and rows atomically."""

    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def migrate_legacy(root: Path) -> dict[str, int]:
    """Journal each intent before moving. Empty/invalid/conflicting files stay put."""
    journal = root / "migration/legacy.json"
    rows = json.loads(journal.read_text()) if journal.is_file() else {}
    events = journal.with_suffix(".jsonl")
    if events.is_file():
        with events.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # An interrupted final append cannot authorize a move.
                rows[row["original_path"]] = row

    def record_intent(row: Record) -> None:
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    sources = sorted(root.glob("*.png"))
    # Persist the entire original-path inventory before moving the first file.
    for source in sources:
        entry = rows.setdefault(str(source), dict(original_path=str(source), status="unresolved"))
        try:
            stat = source.stat()
            entry.setdefault("original_size_bytes", stat.st_size)
            entry.setdefault("original_mtime_ns", stat.st_mtime_ns)
        except OSError as exc:
            entry["inventory_error"] = str(exc)
    write_json(journal, rows)
    for source in sources:
        original = str(source)
        obj = safe_component(source.name.split("_", 1)[0])
        dest = root / "aa/objects" / obj / source.name
        prior = rows.get(original, {})
        row = dict(original_path=original, destination=str(dest), status="unresolved")
        row.update(
            {
                k: prior[k]
                for k in ("original_size_bytes", "original_mtime_ns", "inventory_error")
                if k in prior
            }
        )
        source_valid = False
        try:
            data = png_bytes(source)
            sha = hashlib.sha256(data).hexdigest()
            row["sha256"] = sha
            source_valid = True
            if dest.exists():
                if digest(dest) != sha:
                    row["status"] = "conflict"
                elif prior.get("status") == "move_pending" and source.samefile(dest):
                    source.unlink()
                    row["status"] = "moved"
                    record_intent(row)
                else:
                    # A preexisting duplicate is not an interrupted move.
                    row["status"] = "duplicate_preserved"
            else:
                row["status"] = "move_pending"
                rows[original] = row
                record_intent(row)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, dest)  # Exclusive: a concurrent destination wins.
                if digest(dest) != sha:
                    row["status"] = "conflict"
                    row["detail"] = "Source changed during migration; both paths preserved"
                else:
                    source.unlink()
                    row["status"] = "moved"
                record_intent(row)
        except (OSError, ValueError, zlib.error, EOFError) as exc:
            row["detail"] = str(exc)
            # A recovered archive PNG can resolve an empty legacy placeholder
            # without deleting or rewriting the preserved original entry.
            try:
                recovered_sha = hashlib.sha256(png_bytes(dest)).hexdigest()
                if not source_valid:
                    row["sha256"] = recovered_sha
                    row["status"] = "recovered_original_preserved"
                elif recovered_sha != row["sha256"]:
                    row["status"] = "conflict"
            except (OSError, ValueError, zlib.error, EOFError):
                pass
        rows[original] = row
    for row in rows.values():
        if (
            row["status"] == "move_pending"
            and not Path(row["original_path"]).exists()
            and Path(row["destination"]).is_file()
            and digest(Path(row["destination"])) == row["sha256"]
        ):
            row["status"] = "moved"
    write_json(journal, rows)
    return dict(Counter(row["status"] for row in rows.values()))


def fetch_selected(
    config: WorksetConfig, sources: list[Source], staging: Path
) -> tuple[dict[str, Path], list[Record]]:
    """Transfer only an explicit path allowlist; keep rsync partials for retry."""
    if not sources:
        return {}, []
    paths = sorted({s["path"] for s in sources})
    staging.mkdir(parents=True, exist_ok=True)
    rsync = (
        str(Path("/opt/homebrew/bin/rsync"))
        if Path("/opt/homebrew/bin/rsync").exists()
        else shutil.which("rsync")
    )
    if not rsync:
        return {}, [{"reason": "rsync_unavailable"}]
    args = [
        rsync,
        "-rt",
        "--ignore-times",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--files-from=-",
        "--stats",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=15",
        f"{config.remote}:/",
        str(staging) + "/",
    ]
    issues = []
    deadline = time.monotonic() + config.transfer_timeout
    staged_size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())

    def run(args: Sequence[str], file_list: str) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("bounded workset transfer", config.transfer_timeout)
        return run_command(args, input_text=file_list, timeout=remaining, env={**os.environ, "LC_ALL": "C"})

    for offset in range(0, len(paths), config.transfer_batch_size):
        batch = paths[offset : offset + config.transfer_batch_size]
        file_list = "".join(p.lstrip("/") + "\n" for p in batch)
        batch_paths = [staging / p.lstrip("/") for p in batch]
        before_size = sum(p.stat().st_size for p in batch_paths if p.is_file())
        try:
            try:
                preflight_args = [a for a in args if a != "--ignore-times"]
                preflight = run([preflight_args[0], "--dry-run", *preflight_args[1:]], file_list)
            except subprocess.CalledProcessError as exc:
                if exc.returncode not in {23, 24}:
                    raise
                preflight = subprocess.CompletedProcess(
                    exc.cmd, exc.returncode, exc.stdout or "", exc.stderr or ""
                )
                issues.append({"reason": "remote_source_unavailable", "detail": error_detail(exc)})
            match = re.search(r"Total (?:transferred )?file size:\s*([\d,]+)", preflight.stdout)
            if not match:
                raise ValueError("No transfer size returned by rsync")
            size = int(match.group(1).replace(",", ""))
            pending_match = re.search(r"Total transferred file size:\s*([\d,]+)", preflight.stdout)
            pending = int(pending_match.group(1).replace(",", "")) if pending_match else size
            # Reserve publication space for already staged files and this batch.
            if shutil.disk_usage(staging).free < staged_size * 2 + size * 2 + pending + 100 * 1024 * 1024:
                raise ValueError("Insufficient free space for staging and PNG publication")
            run(args, file_list)
        except subprocess.CalledProcessError as exc:
            issues.append({"reason": "remote_transfer_incomplete", "detail": error_detail(exc)})
            if exc.returncode not in {23, 24}:
                break
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            issues.append({"reason": "remote_transfer_incomplete", "detail": str(exc)[:500]})
            break
        staged_size += sum(p.stat().st_size for p in batch_paths if p.is_file()) - before_size
        emit_progress(
            config, f"Processed remote paths {min(offset + len(batch), len(paths)):,}/{len(paths):,}..."
        )

    return {p: staging / p.lstrip("/") for p in paths if (staging / p.lstrip("/")).is_file()}, issues


def reveal_rows(objects: list[Record], rows: list[Record], scores: ScoreLookup) -> list[Record]:
    """Build score-context records for the selected image identities."""

    by_object = {o["object_id"]: o for o in objects}
    return [
        dict(
            image_id=r["image_id"],
            object_id=r["object_id"],
            cs_score=scores.get(r["filename"], ("", ""))[0],
            coc_score=scores.get(r["filename"], ("", ""))[1],
            object_cs_score=by_object[r["object_id"]]["object_cs_score"],
        )
        for r in rows
    ]


def narrow_aa_workset(manifest: Path, context: int = 3) -> Record:
    """Derive a smaller frozen AA window, writing metadata only.

    Every smaller neighbor window is contained in the original complete window.
    Preserve its image identities and availability receipts; review validation
    and subsequent preparation independently recheck the actual image files.
    """
    manifest = Path(manifest).resolve()
    parent = json.loads(manifest.read_text())
    root = Path(parent["root"])
    lock = root / ".workset-aa.lock"
    lock.mkdir()
    try:
        parent = json.loads(manifest.read_text())
        if (
            parent.get("schema_version") != 1
            or parent.get("project") != "aa"
            or not parent.get("selection_complete")
            or type(context) is not int
            or not 0 <= context < parent["parameters"]["context"]
        ):
            raise ValueError("Narrowing requires a complete AA selection and a smaller nonnegative context")
        if parent["parameters"].get("observation_deduplication") != AA_DEDUPLICATION:
            raise ValueError(
                "Regenerate legacy AA worksets before narrowing: duplicate variants consumed context slots"
            )
        inventory_path = manifest.parent / parent["inventory"]
        if digest(inventory_path) != parent["inventory_sha256"]:
            raise ValueError("Saved source inventory checksum mismatch")
        params = {**parent["parameters"], "context": context}
        fingerprint = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        run_id = parent["run_id"].rsplit("-", 1)[0] + "-" + fingerprint[:16]
        workdir = root / "aa/worksets" / run_id
        if (workdir / "workset.json").exists():
            existing = json.loads((workdir / "workset.json").read_text())
            if existing["fingerprint"] != fingerprint:
                raise ValueError("Destination fingerprint conflict")
            return {
                **existing["summary"],
                "copied": 0,
                "reused": existing["summary"]["availability"].get("available", 0),
            }

        def read(name: str) -> list[Record]:
            with (manifest.parent / parent[name]).open(newline="") as stream:
                return list(csv.DictReader(stream))

        objects, original = read("objects"), read("images")
        groups = defaultdict(list)
        for row in original:
            if not row["filename"].endswith(".png") or not row["relative_path"].endswith(".png"):
                raise ValueError("AA worksets must reference ordinary PNG images")
            groups[row["object_id"]].append(row)
        rows = []
        for obj in objects:
            ordered = sorted(groups[obj["object_id"]], key=lambda r: int(r["order"]))
            chosen: set[int] = set()
            for i, row in enumerate(ordered):
                if row["membership"] == "target":
                    chosen.update(range(max(0, i - context), min(len(ordered), i + context + 1)))
            selected = [dict(ordered[i], order=order) for order, i in enumerate(sorted(chosen))]
            obj["image_count"] = len(selected)
            obj["available_count"] = sum(r["availability"] == "available" for r in selected)
            rows.extend(selected)
        ids = {r["image_id"] for r in rows}
        reveals = [r for r in read("reveal_scores") if r["image_id"] in ids]
        if len(ids) != len(rows) or len(reveals) != len(rows) or {r["image_id"] for r in reveals} != ids:
            raise ValueError("Invalid image/reveal identity join")
        keys = {(r["object_id"], r["filename"]) for r in rows}
        inventory_rows = [
            r for r in json.loads(inventory_path.read_text()) if (r["object_id"], r["filename"]) in keys
        ]
        if len(inventory_rows) != len(rows):
            raise ValueError("Incomplete saved source inventory")
        names = {r["filename"] for r in rows}
        issues = [
            i
            for i in json.loads((manifest.parent / "issues.json").read_text())
            if i["reason"] != "remote_transfer_incomplete" and (not i.get("image") or i["image"] in names)
        ]
        counts = dict(Counter(r["availability"] for r in rows))
        result = dict(
            project="aa",
            run_id=run_id,
            objects=len(objects),
            images=len(rows),
            targets=sum(r["membership"] == "target" for r in rows),
            destination=str(workdir),
            status="complete" if not issues and counts.get("available", 0) == len(rows) else "partial",
            copied=0,
            reused=counts.get("available", 0),
            staging_files_removed=0,
            availability=counts,
            issues=len(issues),
            issue_counts=dict(Counter(i["reason"] for i in issues)),
            migration=parent["summary"].get("migration"),
        )
        workdir.mkdir(parents=True, exist_ok=True)
        write_csv(workdir / "objects.csv", OBJECT_FIELDS, objects)
        write_csv(workdir / "images.csv", IMAGE_FIELDS, rows)
        write_csv(workdir / "reveal_scores.csv", REVEAL_FIELDS, reveals)
        write_json(workdir / "inventory.json", inventory_rows)
        write_json(workdir / "issues.json", issues)
        now = datetime.now(UTC).isoformat()
        write_json(
            workdir / "workset.json",
            {
                **parent,
                "run_id": run_id,
                "fingerprint": fingerprint,
                "parameters": params,
                "inventory": "inventory.json",
                "inventory_sha256": digest(workdir / "inventory.json"),
                "created_at": now,
                "updated_at": now,
                "summary": result,
                "derived_from": dict(
                    workset=str(manifest),
                    manifest_sha256=digest(manifest),
                    context=parent["parameters"]["context"],
                ),
            },
        )
        return result
    finally:
        lock.rmdir()


def prepare_workset(config: WorksetConfig) -> Record:
    """Select, validate, and prepare a resumable candidate workset."""

    emit_progress(config, "Selecting workset candidates...")
    objects, targets, scores, issues = select_candidates(config)
    params = dict(
        project=config.project,
        cutoff=config.cutoff,
        context=config.context,
        scores_sha256=digest(config.scores),
        training_sha256=digest(config.training) if config.training else None,
        roots=[str(p.resolve()) for p in config.source_roots],
        indexes=[str(p.resolve()) for p in config.source_indexes],
        data_root=str(config.data_root.resolve()),
        remote=config.remote,
        remote_index=config.remote_index,
        training_null_is_nontraining=config.training_null_is_nontraining,
        schema_version=1,
    )
    if config.remote_path_maps:
        params["remote_path_maps"] = config.remote_path_maps
    if config.project == "aa":
        params["observation_deduplication"] = AA_DEDUPLICATION
    fingerprint = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    run_id = safe_component(config.label) + "-" + fingerprint[:16]
    workdir = config.destination / config.project / "worksets" / run_id
    existing_manifest = workdir / "workset.json"
    existing = json.loads(existing_manifest.read_text()) if existing_manifest.is_file() else None
    emit_progress(config, f"Indexing archive sources for {len(objects):,} objects...")
    inventory_path = workdir / "inventory.json"
    if (
        config.project == "aa"
        and existing
        and existing.get("selection_complete")
        and existing.get("inventory_sha256")
        and inventory_path.is_file()
    ):
        if digest(inventory_path) != existing["inventory_sha256"]:
            raise ValueError("Saved source inventory checksum mismatch")
        found: Inventory = defaultdict(lambda: defaultdict(list))
        for entry in json.loads(inventory_path.read_text()):
            found[entry["object_id"]][entry["filename"]] = entry["sources"]
            for alias in entry.get("aliases", []):
                found[entry["object_id"]][alias["filename"]] = alias["sources"]
        inventory_issues: list[Record] = []
    else:
        found, inventory_issues = inventory(config, targets)
    issues.extend(inventory_issues)
    inventory_complete = bool(existing and existing.get("selection_complete")) or not inventory_issues
    rows = []
    # Freeze the original membership for a resumed partial run. A failed remote
    # inventory has not established archive neighbors yet and cannot freeze it.
    frozen: dict[str, list[Record]] = {}
    if existing and existing.get("selection_complete"):
        with (workdir / "images.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                frozen.setdefault(row["object_id"], []).append(row)
    aliases = defaultdict(list)
    for objrow in objects:
        obj = objrow["object_id"]
        if config.project == "aa":
            # Deduplicate before counting neighbors, and retain target membership
            # even when an unscored pI replaces a qualifying p2 variant.
            candidates = set(found[obj]) if inventory_complete else set()
            representatives = observation_representatives(candidates, targets[obj], found[obj])
            targets[obj] = {representatives[n] for n in targets[obj]}
            objrow["best_image"] = representatives.get(objrow["best_image"], objrow["best_image"])
            objrow["qualifying_count"] = len(targets[obj])
            for name, representative in sorted(representatives.items()):
                if name != representative:
                    aliases[(obj, representative)].append(dict(filename=name, sources=found[obj][name]))
        names = (
            [r["filename"] for r in frozen[obj]]
            if obj in frozen
            else (
                context_window(set(found[obj]), targets[obj], cast(int, config.context), found[obj])
                if config.project == "aa" and inventory_complete
                else sorted(targets[obj])
            )
        )
        for order, name in enumerate(names):
            safe_component(name)
            relative = f"aa/objects/{obj}/{name}" if config.project == "aa" else ""
            sources = found[obj][name]
            source_ref = sources[0] if sources else None
            reference = (
                ((str(config.remote) + ":" if source_ref["remote"] else "") + source_ref["path"])
                if source_ref
                else ""
            )
            rows.append(
                dict(
                    image_id=image_key(config.project, obj, name),
                    object_id=obj,
                    review_object_name=obj,
                    object_class=objrow["object_class"],
                    product="active_asteroids" if config.project == "aa" else "",
                    filename=name,
                    order=order,
                    membership="target" if name in targets[obj] else "context",
                    relative_path=relative,
                    source_path=reference,
                    availability="missing",
                    sha256="",
                )
            )
    summary = dict(
        project=config.project,
        run_id=run_id,
        objects=len(objects),
        images=len(rows),
        targets=sum(r["membership"] == "target" for r in rows),
        issues=len(issues),
        destination=str(workdir),
    )
    if config.preview:
        return {
            **summary,
            "status": "preview",
            "issue_counts": dict(Counter(i["reason"] for i in issues)),
            "issue_examples": issues[:10],
            "writes": False,
        }
    config.destination.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    migration = None
    lock = config.destination / f".workset-{config.project}.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Another workset writer or interrupted run holds {lock}; check before removing it"
        ) from None
    try:
        # Persist the exact membership before migration, validation, or transfer.
        # A retry therefore cannot silently choose different neighbors.
        write_json(
            inventory_path,
            [
                dict(
                    object_id=r["object_id"],
                    filename=r["filename"],
                    sources=found[r["object_id"]][r["filename"]],
                    **(
                        {"aliases": aliases[(r["object_id"], r["filename"])]}
                        if aliases[(r["object_id"], r["filename"])]
                        else {}
                    ),
                )
                for r in rows
            ],
        )
        write_csv(workdir / "objects.csv", OBJECT_FIELDS, objects)
        write_csv(workdir / "images.csv", IMAGE_FIELDS, rows)
        write_csv(workdir / "reveal_scores.csv", REVEAL_FIELDS, reveal_rows(objects, rows, scores))
        write_json(workdir / "issues.json", issues)
        write_json(
            existing_manifest,
            dict(
                schema_version=1,
                project=config.project,
                run_id=run_id,
                root=str(config.destination.resolve()),
                source_roots=[str(p.resolve()) for p in config.source_roots],
                objects="objects.csv",
                images="images.csv",
                reveal_scores="reveal_scores.csv",
                fingerprint=fingerprint,
                parameters=params,
                selection_complete=inventory_complete,
                inventory="inventory.json",
                inventory_sha256=digest(inventory_path),
                source_files=dict(
                    scores=str(config.scores.resolve()),
                    training=str(config.training.resolve()) if config.training else None,
                ),
                **(
                    {"derived_from": existing["derived_from"]}
                    if existing and "derived_from" in existing
                    else {}
                ),
                created_at=existing.get("created_at") if existing else datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                summary={**summary, "status": "partial", "phase": "preparing"},
            ),
        )
        if config.migrate and config.project == "aa":
            emit_progress(config, "Journaling legacy image migration...")
            migration = migrate_legacy(config.destination)
        emit_progress(config, f"Validating {len(rows):,} selected images...")
        copied = reused = staged_removed = 0
        needs_remote: list[Source] = []
        valid_sources = {}

        def inspect_candidate(row: Record) -> tuple[dict[str, Path], list[Record], list[Source]]:
            obj, name = row["object_id"], row["filename"]
            options = list(found[obj][name])
            if config.project == "aa":
                for root in config.source_roots:
                    for candidate in (root / name, root / (name + ".gz"), root / obj / name):
                        options.append({"path": str(candidate), "remote": False})
                dest = config.destination / row["relative_path"]
                options.insert(0, {"path": str(dest), "remote": False})
            valid: dict[str, Path] = {}
            errors: list[Record] = []
            checked_paths: set[Path] = set()
            for source in options:
                if source["remote"]:
                    continue
                path = Path(source["path"])
                if path in checked_paths:
                    continue
                checked_paths.add(path)
                try:
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                    sha = hashlib.sha256(png_bytes(path)).hexdigest()
                    valid.setdefault(sha, path)
                except (OSError, ValueError, zlib.error, EOFError) as exc:
                    errors.append({"image": name, "reason": "invalid_source_png", "detail": str(exc)})
            return valid, errors, options

        # Keep at most 256 pending reads, with four readers for mounted archives.
        # All publication and metadata mutation remains serialized below.
        with ThreadPoolExecutor(max_workers=4) as readers:
            for offset in range(0, len(rows), 256):
                batch = rows[offset : offset + 256]
                for row, (valid, errors, options) in zip(
                    batch, readers.map(inspect_candidate, batch), strict=False
                ):
                    issues.extend(errors)
                    if len(valid) > 1:
                        row["availability"] = "conflict"
                        issues.append({"image": row["filename"], "reason": "conflicting_image_content"})
                    elif valid:
                        sha, path = next(iter(valid.items()))
                        valid_sources[row["image_id"]] = (path, sha)
                    elif config.project == "aa":
                        needs_remote.extend(s for s in options if s["remote"])
                if offset % 2560 == 0 or offset + 256 >= len(rows):
                    emit_progress(
                        config,
                        f"Validated {min(offset + 256, len(rows)):,}/{len(rows):,} image candidates...",
                    )
        emit_progress(config, f"Fetching {len(needs_remote):,} missing source paths...")
        fetched, fetch_issues = (
            fetch_selected(config, needs_remote, workdir / ".staging")
            if config.project == "aa" and config.remote and needs_remote
            else ({}, [])
        )
        issues.extend(fetch_issues)
        verified_remote = {}
        for row_number, row in enumerate(rows, 1):
            if row_number % 2500 == 0:
                emit_progress(config, f"Published/reused {row_number:,}/{len(rows):,} image candidates...")
            if row["availability"] == "conflict":
                continue
            selected = valid_sources.get(row["image_id"])
            if selected is None:
                remote_valid: dict[str, Path] = {}
                for source in found[row["object_id"]][row["filename"]]:
                    fetched_path = fetched.get(source["path"]) if source["remote"] else None
                    if fetched_path:
                        try:
                            sha = hashlib.sha256(png_bytes(fetched_path)).hexdigest()
                            remote_valid.setdefault(sha, fetched_path)
                            verified_remote[fetched_path] = sha
                        except (OSError, ValueError, zlib.error, EOFError) as exc:
                            issues.append(
                                {
                                    "image": row["filename"],
                                    "reason": "invalid_fetched_png",
                                    "detail": str(exc),
                                }
                            )
                if len(remote_valid) > 1:
                    row["availability"] = "conflict"
                    issues.append({"image": row["filename"], "reason": "conflicting_remote_content"})
                elif remote_valid:
                    sha, path = next(iter(remote_valid.items()))
                    selected = path, sha
            if selected is None:
                continue
            selected_source, sha = selected
            if config.project == "aa":
                dest = config.destination / row["relative_path"]
                if dest.exists():
                    if digest(dest) != sha:
                        row["availability"] = "conflict"
                        issues.append({"image": row["filename"], "reason": "destination_conflict"})
                        continue
                    reused += 1
                else:
                    data = png_bytes(selected_source)
                    if hashlib.sha256(data).hexdigest() != sha:
                        row["availability"] = "conflict"
                        issues.append({"image": row["filename"], "reason": "source_changed_during_run"})
                        continue
                    if (
                        shutil.disk_usage(dest.parent if dest.parent.exists() else config.destination).free
                        < len(data) + 1024 * 1024
                    ):
                        issues.append({"image": row["filename"], "reason": "insufficient_disk_space"})
                        continue
                    publish_png(dest, data)
                    if digest(dest) != sha:
                        raise RuntimeError("Published checksum mismatch")
                    copied += 1
                original_remote = next(
                    (
                        s["path"]
                        for s in found[row["object_id"]][row["filename"]]
                        if s["remote"] and fetched.get(s["path"]) == selected_source
                    ),
                    None,
                )
                row["source_path"] = (
                    f"{config.remote}:{original_remote}" if original_remote else str(selected_source)
                )
            else:
                # Match compatibility identity from the physical product/class/object tree.
                parts = selected_source.parts
                try:
                    i = next(
                        i
                        for i, p in enumerate(parts)
                        if p in {"preliminary_visit_image", "cutouts", "deep_coadd"}
                    )
                    row["product"], row["object_class"] = parts[i], parts[i + 1]
                    row["review_object_name"] = parts[i + 2]
                    if parts[i + 2].replace(" ", "_") != row["object_id"].replace(" ", "_"):
                        raise ValueError("RCC object directory does not match score object")
                except (StopIteration, IndexError, ValueError):
                    issues.append({"image": row["filename"], "reason": "unresolved_rcc_identity"})
                    continue
                row["source_path"] = str(selected_source.resolve())
                row["relative_path"] = "/".join(parts[i:])
            row["sha256"], row["availability"] = sha, "available"
            # Only discard our own verified staging copies after publication.
            # Archive originals, legacy images, and conflicts are never removed.
            if config.project == "aa":
                for option in found[row["object_id"]][row["filename"]]:
                    staged = fetched.get(option["path"]) if option["remote"] else None
                    if staged and verified_remote.get(staged) == sha and staged.exists():
                        staged.unlink()
                        staged_removed += 1
        if config.migrate and config.project == "aa":
            migration = migrate_legacy(config.destination)
        totals = Counter(r["object_id"] for r in rows)
        available = Counter(r["object_id"] for r in rows if r["availability"] == "available")
        for obj in objects:
            obj["image_count"] = totals[obj["object_id"]]
            obj["available_count"] = available[obj["object_id"]]
        reveals = reveal_rows(objects, rows, scores)
        selection_complete = bool(existing and existing.get("selection_complete")) or not any(
            i["reason"]
            in {
                "remote_inventory_failed",
                "missing_source_index",
                "missing_source_root",
                "unreadable_source_directory",
            }
            for i in issues
        )
        counts = dict(Counter(r["availability"] for r in rows))
        status = (
            "complete"
            if selection_complete and not issues and counts.get("available", 0) == len(rows)
            else "partial"
        )
        result = {
            **summary,
            "status": status,
            "copied": copied,
            "reused": reused,
            "staging_files_removed": staged_removed,
            "availability": counts,
            "issues": len(issues),
            "issue_counts": dict(Counter(i["reason"] for i in issues)),
            "migration": migration,
        }
        write_csv(workdir / "objects.csv", OBJECT_FIELDS, objects)
        write_csv(workdir / "images.csv", IMAGE_FIELDS, rows)
        write_csv(workdir / "reveal_scores.csv", REVEAL_FIELDS, reveals)
        write_json(workdir / "issues.json", issues)
        write_json(
            existing_manifest,
            dict(
                schema_version=1,
                project=config.project,
                run_id=run_id,
                root=str(config.destination.resolve()),
                source_roots=[str(p.resolve()) for p in config.source_roots],
                objects="objects.csv",
                images="images.csv",
                reveal_scores="reveal_scores.csv",
                fingerprint=fingerprint,
                parameters=params,
                selection_complete=selection_complete,
                inventory="inventory.json",
                inventory_sha256=digest(inventory_path),
                source_files=dict(
                    scores=str(config.scores.resolve()),
                    training=str(config.training.resolve()) if config.training else None,
                ),
                **(
                    {"derived_from": existing["derived_from"]}
                    if existing and "derived_from" in existing
                    else {}
                ),
                created_at=existing.get("created_at") if existing else datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                summary=result,
            ),
        )
        return result
    finally:
        lock.rmdir()
