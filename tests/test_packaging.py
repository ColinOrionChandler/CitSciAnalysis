"""Checks that run against a wheel installed outside the repository."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import citsci_analysis


def test_distribution_metadata_and_exports() -> None:
    dist = importlib.metadata.distribution("citsci-analysis")
    # Rebuilding an editable checkout can update _version.py without refreshing
    # its installed metadata. Wheels and sdists must agree exactly.
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    if not direct_url.get("dir_info", {}).get("editable", False):
        assert citsci_analysis.__version__ == dist.version
    assert citsci_analysis.__version__ != "0.0.0+unknown"
    assert dist.metadata["Requires-Python"] == ">=3.11"
    assert dist.metadata["License-Expression"] == "BSD-3-Clause"
    assert Path(citsci_analysis.__file__).with_name("py.typed").is_file()
    for name in (
        "ConsolidationResult",
        "DEFAULT_COC_SCORES_PATH",
        "RubinScoringConfig",
        "ScoringConfig",
        "compare_score_files",
        "consolidate_score_directory",
        "discover_score_csvs",
        "inspect_inputs",
        "load_classifications",
        "score_rubin_chunks",
        "score_rubin_file",
        "score_classifications",
        "write_rubin_outputs",
        "write_score_outputs",
    ):
        assert getattr(citsci_analysis, name) is not None
    assert any(
        ep.name == "citsci-score" and ep.value == "citsci_analysis.cli:main" for ep in dist.entry_points
    )


@pytest.mark.parametrize("invocation", ["script", "module"])
def test_installed_cli_scores_csv(invocation: str, tmp_path: Path) -> None:
    input_file = tmp_path / "votes.csv"
    input_file.write_text(
        "username,thumbname,value,train\nsynthetic,training,yes,true\nsynthetic,target,yes,false\n"
    )
    command = (
        [str(Path(sys.executable).with_name("citsci-score"))]
        if invocation == "script"
        else [sys.executable, "-m", "citsci_analysis"]
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run = subprocess.run(
        [
            *command,
            "score",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--label",
            "synthetic",
            "--no-workset",
            "--no-coc-scores",
            "--progress",
            "never",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(run.stdout)
    assert result["diagnostics"]["thumbnails"] == 2
    assert (tmp_path / "outputs/output_total_synthetic.csv").is_file()


def test_installed_consolidation_runs_directly(tmp_path: Path) -> None:
    script = Path(citsci_analysis.__file__).with_name("consolidate.py")
    (tmp_path / "scores.csv").write_text("ThumbName,Confidence\ntarget.png,0.75\n")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--progress", "never"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["output_path"]
    assert "target" in (tmp_path / "consolidated_scores.csv").read_text()


def test_unbuilt_source_tree_entry_paths(tmp_path: Path) -> None:
    """A clean Git archive can run without generated versions or an editable install."""
    import tarfile

    checkout = Path(__file__).resolve().parents[1]
    if not (checkout / ".git").exists():
        pytest.skip("Source checkout check requires Git metadata")
    archive_path = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "HEAD", "-o", str(archive_path)], cwd=checkout, check=True)
    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(source, filter="data")
    env = dict(os.environ, PYTHONPATH=str(source / "src"))
    version = subprocess.run(
        [sys.executable, "-c", "import citsci_analysis as c; print(c.__version__); print(c.__file__)"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert version == ["0.0.0+unknown", str(source / "src/citsci_analysis/__init__.py")]
    subprocess.run(
        [sys.executable, "-m", "citsci_analysis", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=True,
    )
    scores = tmp_path / "source_scores"
    scores.mkdir()
    (scores / "input.csv").write_text("ThumbName,Confidence\ntarget.png,0.75\n")
    subprocess.run(
        [
            sys.executable,
            str(source / "src/citsci_analysis/consolidate.py"),
            str(scores),
            "--progress",
            "never",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=True,
    )
    assert "target" in (scores / "consolidated_scores.csv").read_text()
