from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import struct
import subprocess
import time
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from citsci_analysis import worksets as w
from citsci_analysis.cli import main


def png(color: int = 0) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        w.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes([0, color, 0, 0])))
        + chunk(b"IEND", b"")
    )


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def config(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> w.WorksetConfig:
    source = tmp_path / "source"
    source.mkdir()
    scores = write(
        tmp_path / "scores.csv",
        rows
        or [
            dict(
                thumbname="A_20880105-00000_5",
                score=0.473,
                train=False,
                **{"class": "synthetic"},
                coc_score=5,
            )
        ],
    )
    return w.WorksetConfig(
        "aa", scores, tmp_path / "out", source_roots=(source,), remote=None, context=2, label="test"
    )


def test_threshold_training_and_context(tmp_path: Path) -> None:
    c = config(
        tmp_path,
        [
            dict(thumbname=f"{obj}_20880105-00000_5", score=score, train=train, **{"class": "synthetic"})
            for obj, score, train in [
                ("A", 0.473, False),
                ("B", 0.4729, False),
                ("C", 1, True),
                ("D", 0.9, ""),
            ]
        ],
    )
    objects, targets, lookup, issues = w.select_candidates(c)
    assert [o["object_id"] for o in objects] == ["A"]
    assert issues == [dict(image="D_20880105-00000_5.png", reason="missing_training")]
    names = {f"A_208801{i:02d}-00000_{i}.png" for i in range(1, 10)}
    assert len(w.context_window(names, {"A_20880101-00000_1.png", "A_20880104-00000_4.png"}, 2)) == 6
    assert w.context_window(
        {"A_20880101-00000_b.png", "A_20880101-00000_a.png"}, {"A_20880101-00000_b.png"}, 1
    ) == ["A_20880101-00000_b.png"]


def test_rcc_excludes_training_before_top_three(tmp_path: Path) -> None:
    c = config(tmp_path)
    rows = [
        dict(review_unit_id=f"A_{i}", object_id="A", score=s) for i, s in enumerate([1, 0.6, 0.5, 0.4, 0.1])
    ]
    rows += [dict(review_unit_id="B_1", object_id="B", score=0.473)]
    scores = write(tmp_path / "rcc.csv", rows)
    training = write(
        tmp_path / "train.csv",
        [dict(thumbname=r["review_unit_id"], train=i == 0) for i, r in enumerate(rows)],
    )
    c = replace(c, project="rcc", scores=scores, training=training)
    objects, _, _, _ = w.select_candidates(c)
    assert objects[0]["object_cs_score"] == pytest.approx(0.5)
    assert objects[1]["object_cs_score"] == 0.473
    result = w.prepare_workset(c)
    assert result["copied"] == 0 and result["availability"] == {"missing": 5}
    manifest = json.loads((Path(result["destination"]) / "workset.json").read_text())
    assert "remote_path_maps" not in manifest["parameters"]
    assert not (c.destination / "aa/objects").exists()


def test_preview_no_writes_and_training_conflicts(tmp_path: Path) -> None:
    c = config(tmp_path)
    training = write(tmp_path / "train.csv", [dict(thumbname="A_20880105-00000_5", train=True)])
    result = w.prepare_workset(replace(c, training=training, preview=True))
    assert result["objects"] == 0 and result["issue_counts"] == {"conflicting_training": 1}
    assert not c.destination.exists()


def test_copy_merge_compression_resume_and_blank_scores(tmp_path: Path) -> None:
    c = config(tmp_path)
    for i in range(1, 10):
        (c.source_roots[0] / f"A_208801{i:02d}-00000_{i}.png").write_bytes(png())
    (c.source_roots[0] / "A_20880105-00000_5.fits").write_bytes(b"FITS source must stay put")
    duplicate = tmp_path / "other"
    duplicate.mkdir()
    (duplicate / "A_20880105-00000_5.png.gz").write_bytes(gzip.compress(png()))
    c = replace(c, source_roots=(*c.source_roots, duplicate))
    first = w.prepare_workset(c)
    assert first["status"] == "complete" and first["copied"] == 5
    reveal = pd.read_csv(Path(first["destination"]) / "reveal_scores.csv")
    assert reveal.cs_score.notna().sum() == 1 and reveal.coc_score.dropna().tolist() == [5]
    second = w.prepare_workset(c)
    assert second["run_id"] == first["run_id"] and second["copied"] == 0 and second["reused"] == 5
    assert len(list((c.destination / "aa/objects").rglob("*.png"))) == 5
    assert all(p.suffix == ".png" for p in (c.destination / "aa/objects").rglob("*") if p.is_file())
    assert (c.source_roots[0] / "A_20880105-00000_5.fits").read_bytes() == b"FITS source must stay put"


def test_conflict_and_corrupt_png_are_not_overwritten(tmp_path: Path) -> None:
    c = config(tmp_path)
    name = "A_20880105-00000_5.png"
    (c.source_roots[0] / name).write_bytes(png())
    dest = c.destination / "aa/objects/A" / name
    dest.parent.mkdir(parents=True)
    dest.write_bytes(png(255))
    result = w.prepare_workset(c)
    assert result["availability"] == {"conflict": 1} and dest.read_bytes() == png(255)
    dest.write_bytes(b"")
    result = w.prepare_workset(c)
    assert result["availability"] == {"conflict": 1} and dest.read_bytes() == b""
    for data in (b"", png()[:-10], png()[:40] + b"broken"):
        bad = tmp_path / "bad.png"
        bad.write_bytes(data)
        with pytest.raises((ValueError, zlib.error)):
            w.png_bytes(bad)


def test_migration_preserves_empty_and_conflicts_and_resumes(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    good = root / "A_20880101-00000_1.png"
    good.write_bytes(png())
    empty = root / "B_20880101-00000_1.png"
    empty.write_bytes(b"")
    assert w.migrate_legacy(root) == {"moved": 1, "unresolved": 1}
    assert not good.exists() and empty.exists()
    assert w.migrate_legacy(root) == {"moved": 1, "unresolved": 1}


def test_missing_root_and_remote_error_are_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)

    def failed(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired("ssh", 1)

    monkeypatch.setattr(w, "ssh", failed)
    result = w.prepare_workset(replace(c, remote="synthetic", source_roots=(tmp_path / "absent",)))
    assert result["status"] == "partial"
    assert result["issue_counts"]["remote_inventory_failed"] == 1
    assert result["issue_counts"]["missing_source_root"] == 1


def test_remote_fetch_only_selected_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)
    paths = [f"/archive/A_208801{i:02d}-00000_{i}.png.gz" for i in range(1, 10)]
    monkeypatch.setattr(w, "ssh", lambda *a, **k: subprocess.CompletedProcess([], 0, "\n".join(paths), ""))
    received: list[str] = []

    def fetch(
        config: w.WorksetConfig, sources: list[w.Source], staging: Path
    ) -> tuple[dict[str, Path], list[w.Record]]:
        received.extend(s["path"] for s in sources)
        staging.mkdir(parents=True)
        result = {}
        for s in sources:
            path = staging / Path(s["path"]).name
            path.write_bytes(gzip.compress(png()))
            result[s["path"]] = path
        return result, []

    monkeypatch.setattr(w, "fetch_selected", fetch)
    result = w.prepare_workset(replace(c, remote="synthetic"))
    assert result["copied"] == 5
    assert received == paths[2:7]
    with (Path(result["destination"]) / "images.csv").open() as stream:
        assert all(row["source_path"].startswith("synthetic:/archive/") for row in csv.DictReader(stream))


def test_incomplete_transfer_can_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path, None)
    name = "A_20880105-00000_5.png"
    monkeypatch.setattr(w, "ssh", lambda *a, **k: subprocess.CompletedProcess([], 0, "/archive/" + name, ""))
    monkeypatch.setattr(w, "fetch_selected", lambda *a: ({}, [dict(reason="remote_transfer_incomplete")]))
    first = w.prepare_workset(replace(c, remote="synthetic"))
    assert first["status"] == "partial"
    with (Path(first["destination"]) / "images.csv").open() as stream:
        assert next(csv.DictReader(stream))["source_path"] == "synthetic:/archive/" + name
    (c.source_roots[0] / name).write_bytes(png())
    second = w.prepare_workset(replace(c, remote="synthetic"))
    assert second["run_id"] == first["run_id"] and second["status"] == "complete"


def test_cli_defaults_on_and_opt_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    c = config(tmp_path)
    assert (
        main(
            [
                "prepare-workset",
                str(c.scores),
                "--project",
                "aa",
                "--workset-root",
                str(c.destination),
                "--source-root",
                str(c.source_roots[0]),
                "--no-remote",
                "--preview",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "preview"
    assert not c.destination.exists()
    raw = write(
        tmp_path / "raw.csv",
        [
            dict(
                username="u",
                thumbname="A_20880105-00000_5",
                value="yes",
                train=False,
                **{"class": "synthetic"},
            )
        ],
    )
    args = [
        "score",
        str(raw),
        "--progress",
        "never",
        "--output-dir",
        str(tmp_path / "analysis"),
        "--no-coc-scores",
        "--workset-root",
        str(c.destination),
        "--source-root",
        str(c.source_roots[0]),
        "--no-remote",
    ]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["workset"]["project"] == "aa"
    assert captured.err == ""
    assert main([*args, "--no-workset"]) == 0
    assert json.loads(capsys.readouterr().out)["workset"] is None


def test_explicit_nullable_convention_and_negative_training_control(tmp_path: Path) -> None:
    c = config(tmp_path)
    rows = [dict(review_unit_id=f"A_{i}", object_id="A", score=0.9) for i in range(3)]
    path = write(tmp_path / "rcc.csv", rows)
    training = write(
        tmp_path / "train.csv",
        [
            dict(thumbname="A_0", train="", training_subject=""),
            dict(thumbname="A_1", train="0", training_subject=True),
            dict(thumbname="A_2", train="positive", training_subject=True),
        ],
    )
    strict = replace(c, project="rcc", scores=path, training=training)
    assert not w.select_candidates(strict)[0]
    objects, targets, _, issues = w.select_candidates(replace(strict, training_null_is_nontraining=True))
    assert targets == {"A": {"A_0.png"}} and not issues


def test_no_distant_context_when_inventory_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)
    for i in [1, 5, 9]:
        (c.source_roots[0] / f"A_208801{i:02d}-00000_{i}.png").write_bytes(png())

    def fail(*a: Any, **k: Any) -> None:
        raise subprocess.TimeoutExpired("ssh", 1)

    monkeypatch.setattr(w, "ssh", fail)
    result = w.prepare_workset(replace(c, remote="synthetic"))
    assert result["images"] == 1 and result["copied"] == 1 and result["status"] == "partial"


def test_exclusive_publication_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "image.png"
    target.write_bytes(png())
    with pytest.raises(FileExistsError):
        w.publish_png(target, png(100))
    assert target.read_bytes() == png()
    assert list(tmp_path.iterdir()) == [target]


def test_rsync_preflight_does_not_transfer_without_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    calls = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "Total file size: 999999999999999 bytes\n", "")

    monkeypatch.setattr(w, "run_command", run)
    monkeypatch.setattr(shutil, "which", lambda _: "/synthetic/rsync")
    files, issues = w.fetch_selected(
        replace(c, remote="synthetic"), [dict(path="/archive/A.png")], tmp_path / "stage"
    )
    assert not files and len(calls) == 1 and "--dry-run" in calls[0]
    assert issues[0]["reason"] == "remote_transfer_incomplete"


def test_remote_content_conflict_preserves_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)
    name = "A_20880105-00000_5.png"
    monkeypatch.setattr(
        w, "ssh", lambda *a, **k: subprocess.CompletedProcess([], 0, "/one/" + name + "\n/two/" + name, "")
    )

    def fetch(
        config: w.WorksetConfig, sources: list[w.Source], staging: Path
    ) -> tuple[dict[str, Path], list[w.Record]]:
        staging.mkdir(parents=True)
        result = {}
        for i, s in enumerate(sources):
            p = staging / f"{i}.png"
            p.write_bytes(png(i))
            result[s["path"]] = p
        return result, []

    monkeypatch.setattr(w, "fetch_selected", fetch)
    result = w.prepare_workset(replace(c, remote="synthetic"))
    assert result["availability"] == {"conflict": 1} and result["copied"] == 0
    assert len(list((Path(result["destination"]) / ".staging").glob("*.png"))) == 2


def test_timeout_byte_stderr_is_serializable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)

    def fail(*a: Any, **k: Any) -> None:
        raise subprocess.TimeoutExpired("ssh", 1, stderr=b"connection lost")

    monkeypatch.setattr(w, "ssh", fail)
    result = w.prepare_workset(replace(c, remote="synthetic"))
    assert result["status"] == "partial"
    assert (
        json.loads((Path(result["destination"]) / "issues.json").read_text())[0]["detail"]
        == "connection lost"
    )


def test_interrupted_publication_keeps_selection_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    for i in [4, 5, 6]:
        (c.source_roots[0] / f"A_208801{i:02d}-00000_{i}.png").write_bytes(png())
    original = w.publish_png
    calls = []

    def interrupted(path: Path, data: bytes) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("interrupted")
        original(path, data)

    monkeypatch.setattr(w, "publish_png", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        w.prepare_workset(c)
    manifest = next(c.destination.glob("aa/worksets/*/workset.json"))
    assert json.loads(manifest.read_text())["selection_complete"]
    assert json.loads(manifest.read_text())["summary"]["status"] == "partial"
    # New archive neighbors do not change an already frozen selection.
    (c.source_roots[0] / "A_20880103-00000_3.png").write_bytes(png())
    monkeypatch.setattr(w, "publish_png", original)
    result = w.prepare_workset(c)
    assert result["images"] == 3 and result["copied"] == 2 and result["reused"] == 1


def test_source_change_before_publication_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    name = "A_20880105-00000_5.png"
    (c.source_roots[0] / name).write_bytes(png())
    original = w.png_bytes
    reads = []

    def changing(path: Path) -> bytes:
        data = original(path)
        if path.name == name:
            reads.append(path)
            if len(reads) == 2:
                return png(255)
        return data

    monkeypatch.setattr(w, "png_bytes", changing)
    result = w.prepare_workset(c)
    assert result["availability"] == {"conflict": 1}
    assert not (c.destination / "aa/objects/A" / name).exists()


def test_timeout_terminates_command_children(tmp_path: Path) -> None:
    import sys
    import time

    marker = tmp_path / "unexpected-child-output"
    child = (
        f'import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text("leaked")'
    )
    parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(20)'
    with pytest.raises(subprocess.TimeoutExpired):
        w.run_command([sys.executable, "-c", parent], timeout=0.2)
    time.sleep(1.2)
    assert not marker.exists()


def test_resumed_workset_reuses_verified_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    (c.source_roots[0] / "A_20880105-00000_5.png").write_bytes(png())
    first = w.prepare_workset(c)

    def no_scan(*a: Any, **k: Any) -> None:
        raise AssertionError("A frozen selection must not rescan archives")

    monkeypatch.setattr(w, "inventory", no_scan)
    second = w.prepare_workset(c)
    assert first["run_id"] == second["run_id"] and second["copied"] == 0 and second["reused"] == 1
    manifest = Path(first["destination"]) / "workset.json"
    assert json.loads(manifest.read_text())["source_files"]["scores"] == str(c.scores)
    (manifest.parent / "inventory.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        w.prepare_workset(c)


def test_explicit_remote_path_mapping_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = config(tmp_path)
    monkeypatch.setattr(
        w, "ssh", lambda *a, **k: subprocess.CompletedProcess([], 0, "/old/A_20880105-00000_5.png", "")
    )
    received: list[str] = []

    def fetch(
        config: w.WorksetConfig, sources: list[w.Source], staging: Path
    ) -> tuple[dict[str, Path], list[w.Record]]:
        received.extend(s["path"] for s in sources)
        return {}, [dict(reason="remote_transfer_incomplete")]

    monkeypatch.setattr(w, "fetch_selected", fetch)
    result = w.prepare_workset(replace(c, remote="synthetic", remote_path_maps=(("/old", "/current"),)))
    assert received == ["/current/A_20880105-00000_5.png"]
    data = json.loads((Path(result["destination"]) / "workset.json").read_text())
    assert data["parameters"]["remote_path_maps"] == [["/old", "/current"]]


def test_verified_transfer_staging_is_removed_without_touching_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    name = "A_20880105-00000_5.png"
    monkeypatch.setattr(w, "ssh", lambda *a, **k: subprocess.CompletedProcess([], 0, "/archive/" + name, ""))

    def fetch(
        config: w.WorksetConfig, sources: list[w.Source], staging: Path
    ) -> tuple[dict[str, Path], list[w.Record]]:
        staging.mkdir(parents=True)
        path = staging / name
        path.write_bytes(png())
        return {"/archive/" + name: path}, []

    monkeypatch.setattr(w, "fetch_selected", fetch)
    result = w.prepare_workset(replace(c, remote="synthetic"))
    assert result["staging_files_removed"] == 1
    assert not (Path(result["destination"]) / ".staging" / name).exists()
    assert (c.destination / "aa/objects/A" / name).read_bytes() == png()


def test_partial_rsync_preflight_still_transfers_available_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = config(tmp_path)
    calls = []
    staging = tmp_path / "stage"

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if "--dry-run" in args:
            raise subprocess.CalledProcessError(
                23,
                args,
                output="Total file size: 100 bytes\nTotal transferred file size: 100 bytes\n",
                stderr="one indexed source is missing",
            )
        p = staging / "archive/A.png"
        p.parent.mkdir(parents=True)
        p.write_bytes(png())
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(w, "run_command", run)
    monkeypatch.setattr(shutil, "which", lambda _: "/synthetic/rsync")
    files, issues = w.fetch_selected(
        replace(c, remote="synthetic"), [dict(path="/archive/A.png"), dict(path="/archive/B.png")], staging
    )
    assert len(calls) == 2 and "/archive/A.png" in files
    assert "--ignore-times" not in calls[0] and "--ignore-times" in calls[1]
    assert issues[0]["reason"] == "remote_source_unavailable"


def test_remote_batches_share_a_total_deadline_and_keep_completed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = replace(config(tmp_path), remote="synthetic", transfer_batch_size=2, transfer_timeout=10)
    staging = tmp_path / "staging"
    calls = []
    clock = [0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(shutil, "which", lambda _: "/synthetic/rsync")

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        names = kwargs["input_text"].splitlines()
        calls.append((names, kwargs["timeout"]))
        if "--dry-run" in args:
            clock[0] += 1
            return subprocess.CompletedProcess(args, 0, "Total file size: 100 bytes\n", "")
        for name in names:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png())
        clock[0] = 11
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(w, "run_command", run)
    paths: list[w.Source] = [dict(path=f"/archive/{i}.png") for i in range(5)]
    files, issues = w.fetch_selected(c, paths, staging)
    assert len(files) == 2 and len(calls) == 2
    assert calls[0][0] == ["archive/0.png", "archive/1.png"]
    assert calls[1][1] == 9 and issues[-1]["reason"] == "remote_transfer_incomplete"


def test_migration_records_recovered_empty_original_without_deleting_it(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    name = "A_20880101-00000_1.png"
    original = root / name
    original.write_bytes(b"")
    assert w.migrate_legacy(root) == {"unresolved": 1}
    recovered = root / "aa/objects/A" / name
    recovered.parent.mkdir(parents=True)
    recovered.write_bytes(png())
    assert w.migrate_legacy(root) == {"recovered_original_preserved": 1}
    row = json.loads((root / "migration/legacy.json").read_text())[str(original)]
    assert row["destination"] == str(recovered) and row["sha256"] == w.digest(recovered)
    assert original.read_bytes() == b""


def test_interrupted_migration_resumes_exclusive_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = root / "A_20880101-00000_1.png"
    source.write_bytes(png())
    unlink = Path.unlink

    def interrupted(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == source:
            raise OSError("interrupted before removal")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    assert w.migrate_legacy(root) == {"move_pending": 1}
    assert source.exists()
    monkeypatch.setattr(Path, "unlink", unlink)
    assert w.migrate_legacy(root) == {"moved": 1}
    assert not source.exists()


def test_migration_never_overwrites_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = root / "A_20880101-00000_1.png"
    source.write_bytes(png())
    link = os.link

    def concurrent(src: Path, dest: Path) -> None:
        dest.write_bytes(png(255))
        return link(src, dest)

    monkeypatch.setattr(os, "link", concurrent)
    assert w.migrate_legacy(root) == {"conflict": 1}
    assert source.read_bytes() == png()
    assert (root / "aa/objects/A" / source.name).read_bytes() == png(255)


def test_default_arnor_source_uses_verified_archive_mapping(tmp_path: Path) -> None:
    c = replace(config(tmp_path), remote=w.DEFAULT_REMOTE)
    assert c.remote_path_maps == w.DEFAULT_REMOTE_PATH_MAPS
    explicit = replace(c, remote_path_maps=(("/old", "/custom"),))
    assert explicit.remote_path_maps == (("/old", "/custom"),)
    assert replace(c, remote="custom-host", remote_path_maps=()).remote_path_maps == ()


def test_migration_inventory_is_durable_before_first_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    first = root / "A_20880101-00000_1.png"
    first.write_bytes(png())
    second = root / "B_20880101-00000_1.png"
    second.write_bytes(b"")
    link = os.link

    def inspect_before_move(source: Path, dest: Path) -> None:
        inventory = json.loads((root / "migration/legacy.json").read_text())
        assert set(inventory) == {str(first), str(second)}
        assert inventory[str(second)]["original_size_bytes"] == 0
        return link(source, dest)

    monkeypatch.setattr(os, "link", inspect_before_move)
    assert w.migrate_legacy(root) == {"moved": 1, "unresolved": 1}


def test_rcc_resume_resolves_images_that_arrive_in_existing_roots(tmp_path: Path) -> None:
    c = config(tmp_path)
    name = "A_20880101-00000_1"
    scores = write(tmp_path / "rcc.csv", [dict(review_unit_id=name, object_id="A", score=0.8, train=False)])
    c = replace(c, project="rcc", scores=scores)
    first = w.prepare_workset(c)
    assert first["availability"] == {"missing": 1}
    image = c.source_roots[0] / "preliminary_visit_image/synthetic/A" / f"{name}.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(png())
    second = w.prepare_workset(c)
    assert first["run_id"] == second["run_id"]
    assert second["availability"] == {"available": 1} and second["copied"] == 0


def test_project_context_defaults_and_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from citsci_analysis.cli import build_parser, run_workset

    assert w.WorksetConfig("aa", tmp_path / "scores", tmp_path / "out").context == 3
    assert w.WorksetConfig("rcc", tmp_path / "scores", tmp_path / "out").context == 10
    monkeypatch.setattr(w, "prepare_workset", lambda c: c)
    for project, expected in [("aa", 3), ("rcc", 10)]:
        args = build_parser().parse_args(
            [
                "prepare-workset",
                str(tmp_path / "scores"),
                "--project",
                project,
                "--workset-root",
                str(tmp_path / "out"),
            ]
        )
        assert cast(w.WorksetConfig, run_workset(args, args.scores, project)).context == expected
        args.context = 0
        assert cast(w.WorksetConfig, run_workset(args, args.scores, project)).context == 0
        args.context = 10
        assert cast(w.WorksetConfig, run_workset(args, args.scores, project)).context == 10


def test_narrow_workset_keeps_targets_order_scores_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Endpoint, overlapping windows, and an unclassified neighbor.
    names = [f"A_20880101-000{i}0_{i}.png" for i in range(9)]
    c = config(
        tmp_path,
        [
            dict(thumbname=names[i], score=0.8, train=False, **{"class": "synthetic"}, coc_score=7)
            for i in [0, 3, 8]
        ],
    )
    c = replace(c, context=4)
    for name in names:
        (c.source_roots[0] / name).write_bytes(png())
    parent = w.prepare_workset(c)
    path = Path(parent["destination"]) / "workset.json"
    parent_bytes = {p.name: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}

    def forbidden(*a: Any, **k: Any) -> None:
        raise AssertionError("Narrowing must not access archives or copy images")

    monkeypatch.setattr(w, "inventory", forbidden)
    monkeypatch.setattr(w, "fetch_selected", forbidden)
    monkeypatch.setattr(w, "publish_png", forbidden)
    result = w.narrow_aa_workset(path, 1)
    child = Path(result["destination"])
    with (child / "images.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["filename"] for r in rows] == [names[i] for i in [0, 1, 2, 3, 4, 7, 8]]
    assert [int(r["order"]) for r in rows] == list(range(7))
    assert result["copied"] == 0 and result["reused"] == 7 and result["targets"] == 3
    with (path.parent / "images.csv").open() as f:
        old = {r["filename"]: r for r in csv.DictReader(f)}
    assert all(r["image_id"] == old[r["filename"]]["image_id"] for r in rows)
    with (child / "reveal_scores.csv").open() as f:
        reveal = list(csv.DictReader(f))
    assert sum(bool(r["cs_score"]) for r in reveal) == 3
    assert all(p.read_bytes() == parent_bytes[p.name] for p in path.parent.iterdir() if p.is_file())
    assert len(list((c.destination / "aa/objects").rglob("*.png"))) == 9
    before = (child / "workset.json").read_bytes()
    assert w.narrow_aa_workset(path, 1)["copied"] == 0
    assert (child / "workset.json").read_bytes() == before
    # A normal resume resolves to the same derived run without reindexing.
    resumed = w.prepare_workset(replace(c, context=1))
    assert resumed["run_id"] == result["run_id"] and resumed["copied"] == 0
    assert json.loads((child / "workset.json").read_text())["derived_from"]["workset"] == str(path)


def test_narrow_rejects_widening_incomplete_or_rcc_parent(tmp_path: Path) -> None:
    c = config(tmp_path)
    parent = w.prepare_workset(c)
    path = Path(parent["destination"]) / "workset.json"
    original = json.loads(path.read_text())
    with pytest.raises(ValueError):
        w.narrow_aa_workset(path, 3)
    updates_list: list[w.Record] = [dict(project="rcc"), dict(selection_complete=False)]
    for updates in updates_list:
        path.write_text(json.dumps({**original, **updates}))
        with pytest.raises(ValueError):
            w.narrow_aa_workset(path, 1)
    path.write_text(json.dumps(original))
    (path.parent / "inventory.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        w.narrow_aa_workset(path, 1)
    assert not (c.destination / ".workset-aa.lock").exists()


def test_observation_variants_prefer_pi_before_counting_neighbors() -> None:
    names = {
        f"A_20880101-000{i}0_{version}_p{product}.png"
        for i in range(9)
        for version, product in [("1", "2"), ("2", "I"), ("3", "I")]
    }
    target = "A_20880101-00040_1_p2.png"
    chosen = w.context_window(names, {target}, 3)
    assert chosen == [f"A_20880101-000{i}0_2_pI.png" for i in range(1, 8)]
    assert w.context_window(names, {target}, 0) == ["A_20880101-00040_2_pI.png"]
    assert w.observation_representatives(names, {target}) == w.observation_representatives(
        set(reversed(sorted(names))), {target}
    )


def test_observation_source_time_handles_start_midpoint_and_close_exposures() -> None:
    start = "A_20141210-03565_1720764_p2.png"
    midpoint = "A_20141210-03583_1720764_pI.png"
    local = "A_20141210-03565_9999999_p2.png"
    adjacent = "A_20141210-03565_1234567_pI.png"
    unknown = "A_20141210-03565_8888888_p2.png"
    source = "/archive/c4d_141210_040104_ooi_i_v1/Thumbs/PNG/"
    sources: dict[str, list[w.Source]] = {n: [dict(path=source + n, remote=True)] for n in [start, midpoint]}
    names = {start, midpoint, local}
    assert set(w.observation_representatives(names, {start}, sources).values()) == {midpoint}
    sources[adjacent] = [dict(path=source.replace("040104", "040108") + adjacent, remote=True)]
    # Distinct exposures within a ten-second filename bin stay separate;
    # an ambiguous filename-only entry cannot be assigned to either exposure.
    reps = w.observation_representatives(names | {adjacent, unknown}, {start}, sources)
    assert reps[start] == midpoint and reps[adjacent] == adjacent
    assert reps[unknown] not in {midpoint, adjacent}


def test_observation_fallback_keeps_objects_dates_and_unknown_names_separate() -> None:
    names = {
        "A_20190101-23595_1_p2.png",
        "A_20190102-00000_1_pI.png",
        "B_20190102-00000_1_pI.png",
        "A_unknown_1_p2.png",
        "A_unknown_2_pI.png",
        "A_20190230-00000_1_p2.png",
        "A_20190230-00000_2_pI.png",
    }
    assert set(w.observation_representatives(names, set()).values()) == names


def test_prepare_deduplicates_targets_keeps_score_provenance_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p2 = "A_20880105-00000_1_p2.png"
    pi = "A_20880105-00000_2_pI.png"
    c = config(tmp_path, [dict(thumbname=p2, score=0.9, train=False, **{"class": "synthetic"}, coc_score=7)])
    for name in [p2, pi, "A_20880104-00000_1_pI.png", "A_20880106-00000_1_pI.png"]:
        (c.source_roots[0] / name).write_bytes(png())
    first = w.prepare_workset(c)
    assert first["images"] == 3 and first["targets"] == 1
    folder = Path(first["destination"])
    rows = list(csv.DictReader((folder / "images.csv").open()))
    target = next(r for r in rows if r["membership"] == "target")
    assert target["filename"] == pi
    assert target["image_id"] == w.image_key("aa", "A", pi)
    objects = list(csv.DictReader((folder / "objects.csv").open()))
    assert objects[0]["best_image"] == pi and objects[0]["qualifying_count"] == "1"
    reveal = list(csv.DictReader((folder / "reveal_scores.csv").open()))
    # The p2 score must not be presented as a measurement of its unscored pI.
    assert all(r["cs_score"] == "" and r["coc_score"] == "" and r["object_cs_score"] == "0.9" for r in reveal)
    inventory = json.loads((folder / "inventory.json").read_text())
    entry = next(r for r in inventory if r["filename"] == pi)
    assert entry["aliases"][0]["filename"] == p2
    assert (c.source_roots[0] / p2).exists()
    assert not (c.destination / "aa/objects/A" / p2).exists()
    monkeypatch.setattr(w, "inventory", lambda *a: pytest.fail("Frozen selection must not rescan"))
    second = w.prepare_workset(c)
    assert second["run_id"] == first["run_id"] and second["copied"] == 0
    assert json.loads((folder / "inventory.json").read_text()) == inventory
    resumed = list(csv.DictReader((folder / "images.csv").open()))
    assert [{k: v for k, v in r.items() if k != "source_path"} for r in resumed] == [
        {k: v for k, v in r.items() if k != "source_path"} for r in rows
    ]


def test_legacy_fingerprint_cannot_resume_or_narrow_duplicate_selection(tmp_path: Path) -> None:
    c = config(tmp_path)
    result = w.prepare_workset(c)
    path = Path(result["destination"]) / "workset.json"
    manifest = json.loads(path.read_text())
    params = manifest["parameters"].copy()
    assert params.pop("observation_deduplication") == w.AA_DEDUPLICATION
    import hashlib

    legacy = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    assert legacy != manifest["fingerprint"]
    manifest["parameters"] = params
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Regenerate legacy"):
        w.narrow_aa_workset(path, 1)
