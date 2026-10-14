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
    assert citsci_analysis.__version__ == dist.version
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
