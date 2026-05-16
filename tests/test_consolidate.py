from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from citsci_analysis.cli import main
from citsci_analysis.consolidate import consolidate_score_directory


def test_consolidates_headerless_rows_and_keeps_latest_dated_score(tmp_path) -> None:
    input_path = tmp_path / "scores.csv"
    input_path.write_text(
        "\n".join(
            [
                "2026-05-01 20:10:52.835483,0,/tmp/thumb_a.png,2",
                "not-a-date,1,/tmp/thumb_a.jpg,1",
                "2026-05-02 01:00:00,2,/tmp/thumb_a.png,3",
                ",3,/tmp/thumb_b.fits,0",
            ]
        )
    )

    result = consolidate_score_directory(tmp_path)
    output = pd.read_csv(tmp_path / "consolidated_scores.csv", keep_default_na=False)

    assert result.summary["parsed_rows"] == 4
    assert result.summary["malformed_date_rows"] == 1
    assert result.summary["duplicate_rows_removed"] == 2
    assert output["thumbnail_basename"].tolist() == ["thumb_a", "thumb_b"]

    rows = output.set_index("thumbnail_basename")
    assert rows.loc["thumb_a", "datetime"] == "2026-05-02 01:00:00"
    assert rows.loc["thumb_a", "path"] == "/tmp/thumb_a.png"
    assert rows.loc["thumb_a", "score"] == 3.0
    assert rows.loc["thumb_b", "datetime"] == ""
    assert rows.loc["thumb_b", "score"] == 0.0


def test_valid_datetime_beats_undated_duplicate(tmp_path) -> None:
    input_path = tmp_path / "scores.csv"
    input_path.write_text(
        "\n".join(
            [
                ",0,/tmp/same.png,1",
                "2026-05-01 00:00:00,1,/tmp/same.png,2",
            ]
        )
    )

    consolidate_score_directory(tmp_path)
    output = pd.read_csv(tmp_path / "consolidated_scores.csv")

    assert output.loc[0, "thumbnail_basename"] == "same"
    assert output.loc[0, "score"] == 2.0


def test_all_undated_duplicates_keep_last_deterministic_row(tmp_path) -> None:
    (tmp_path / "a.csv").write_text("bad-date,0,/tmp/same.png,1\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.csv").write_text(",0,/tmp/same.png,2\n")

    result = consolidate_score_directory(tmp_path)
    output = pd.read_csv(tmp_path / "consolidated_scores.csv")

    assert result.summary["discovered_files"] == 2
    assert output.loc[0, "thumbnail_basename"] == "same"
    assert output.loc[0, "score"] == 2.0


def test_legacy_thumbname_confidence_csv_drops_extra_columns(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.csv"
    legacy_path.write_text(
        "\n".join(
            [
                "ObjName,HelioDistAu,PeriDistAu,PctPeri,Confidence,deltaMagLim,ThumbName",
                "101P/Chernykh-A,4.80643,2.3504,None,0,-3.9887,0800180C_20180616-04450_1286146_014y_qJ_Ob_300s_T-4_pI_ovX",
                "101P/Chernykh-A,4.80594,2.3504,None,1,-3.9887,0800180C_20180616-07075_1287780_014y_qJ_Ob_300s_T-4_pI_ovX",
            ]
        )
    )

    consolidate_score_directory(tmp_path)
    output = pd.read_csv(tmp_path / "consolidated_scores.csv", keep_default_na=False)

    assert output.columns.tolist() == ["datetime", "path", "thumbnail_basename", "score"]
    assert output["datetime"].tolist() == ["", ""]
    assert output["path"].tolist() == ["", ""]
    assert output["score"].tolist() == [0.0, 1.0]
    assert output["thumbnail_basename"].tolist() == [
        "0800180C_20180616-04450_1286146_014y_qJ_Ob_300s_T-4_pI_ovX",
        "0800180C_20180616-07075_1287780_014y_qJ_Ob_300s_T-4_pI_ovX",
    ]


def test_simple_headered_score_file_and_output_exclusion_on_rerun(tmp_path) -> None:
    first_path = tmp_path / "first.csv"
    first_path.write_text("thumb_name,score\nnested/thumb_c.png,0.5\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.csv").write_text("datetime,path,score\n2026-05-01 01:02:03,/tmp/thumb_d.jpeg,0.75\n")

    first = consolidate_score_directory(tmp_path)
    second = consolidate_score_directory(tmp_path)
    output = pd.read_csv(tmp_path / "consolidated_scores.csv")

    assert first.summary["discovered_files"] == 2
    assert second.summary["discovered_files"] == 2
    assert output["thumbnail_basename"].tolist() == ["thumb_c", "thumb_d"]
    assert output["score"].tolist() == [0.5, 0.75]


def test_consolidate_scores_cli_writes_default_output(tmp_path, capsys) -> None:
    input_path = tmp_path / "scores.csv"
    input_path.write_text("2026-05-01 00:00:00,0,/tmp/cli_thumb.png,1\n")

    status = main(["consolidate-scores", str(tmp_path), "--progress", "always"])

    assert status == 0
    output_path = tmp_path / "consolidated_scores.csv"
    assert output_path.exists()
    captured = capsys.readouterr()
    assert "Found 1 CSV files." in captured.err

    summary = json.loads(captured.out)
    assert summary["parsed_rows"] == 1
    assert summary["unique_basenames"] == 1
    assert summary["output_path"] == str(output_path)


def test_consolidate_script_runs_directly_from_working_directory(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "src" / "citsci_analysis" / "consolidate.py"
    input_path = tmp_path / "scores.csv"
    input_path.write_text("2026-05-01 00:00:00,0,/tmp/direct_thumb.png,1\n")

    completed = subprocess.run(
        [sys.executable, str(script_path), "--progress", "never"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    output_path = tmp_path / "consolidated_scores.csv"
    assert output_path.exists()
    summary = json.loads(completed.stdout)
    assert summary["output_path"] == "consolidated_scores.csv"
    assert summary["unique_basenames"] == 1
