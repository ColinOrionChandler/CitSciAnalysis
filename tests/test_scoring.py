from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from citsci_analysis.cli import main
from citsci_analysis.scoring import (
    SCORE_OUTPUT_COLUMNS,
    ScoringConfig,
    calculate_metric_weight,
    compare_score_files,
    load_classifications,
    load_problematic_training_thumbs,
    normalize_thumbnames,
    row_weights_for_mode,
    score_classifications,
    write_score_outputs,
)


def test_thumbname_normalization_strips_paths_and_extensions() -> None:
    raw = pd.Series(["/tmp/a/thumb.png", "b.fits", "c.fits.fz", "plain"])
    assert normalize_thumbnames(raw).tolist() == ["thumb", "b", "c", "plain"]


def test_metric_omits_missing_terms() -> None:
    weights = calculate_metric_weight(
        train_yes_count=pd.Series([0, 5]),
        train_count=pd.Series([0, 10]),
        test_yes_count=pd.Series([1, 2]),
        test_count=pd.Series([4, 4]),
    )
    assert np.isclose(weights[0], ((2 * np.log10(4) / 4) + (1 * 0.75)) / 3)
    assert np.isclose(weights[1], ((7 * 0.5) + (2 * np.log10(14) / 4) + (1 * 0.5)) / 10)


def test_naive_scoring_uses_static_user_weights_and_dedupes() -> None:
    frame = pd.DataFrame(
        {
            "row_order": range(7),
            "classification_id": range(7),
            "user_key": ["u1", "u1", "u1", "u2", "u2", "u2", "u2"],
            "user_name": ["u1", "u1", "u1", "u2", "u2", "u2", "u2"],
            "thumbname": ["train_active", "test_a", "test_a", "train_active", "test_a", "test_b", "test_b"],
            "answer": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "train": [True, False, False, True, False, False, False],
            "batch": [1, 1, 1, 1, 1, 1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )
    config = ScoringConfig(mode="naive", dedupe_user_thumbnail=True)
    result = score_classifications(frame, config)
    scores = result.scores.set_index("thumbname")["score"]

    assert scores["train_active"] == 1.0
    assert 0.0 <= scores["test_a"] <= 1.0
    assert scores["test_b"] == 0.0
    assert result.diagnostics["thumbnails"] == 3


def test_load_classifications_normalizes_csv_columns_and_user_fallbacks(tmp_path) -> None:
    input_path = tmp_path / "classifications.csv"
    pd.DataFrame(
        {
            "classification_id": [11, 12, 13],
            "username": ["alice", None, ""],
            "user_id": [None, "42", None],
            "user_ip": [None, None, "127.0.0.1"],
            "thumbname": ["/tmp/first.fits.fz", None, "third.png"],
            "filename": [None, "second.fits", None],
            "value": ["yes", "no", "maybe"],
            "train": ["true", "", None],
            "batch": ["1", "2", None],
        }
    ).to_csv(input_path, index=False)

    normalized = load_classifications(input_path)

    assert normalized["thumbname"].tolist() == ["first", "second", "third"]
    assert normalized["user_key"].tolist() == ["alice", "user_id:42", "user_ip:127.0.0.1"]
    assert normalized["answer"].iloc[:2].tolist() == [1.0, 0.0]
    assert np.isnan(normalized["answer"].iloc[2])
    assert normalized["train"].tolist() == [True, False, False]
    assert normalized["class"].isna().all()


def test_load_classifications_preserves_class_column(tmp_path) -> None:
    input_path = tmp_path / "classifications.csv"
    pd.DataFrame(
        {
            "classification_id": [1, 2],
            "username": ["alice", "bob"],
            "thumbname": ["active.png", "quiet.png"],
            "value": ["yes", "no"],
            "class": ["object-a", ""],
        }
    ).to_csv(input_path, index=False)

    normalized = load_classifications(input_path)

    assert normalized["class"].iloc[0] == "object-a"
    assert pd.isna(normalized["class"].iloc[1])


def test_load_problematic_training_thumbs_accepts_csv_variants(tmp_path) -> None:
    input_path = tmp_path / "problematic.csv"
    pd.DataFrame(
        {
            "thumb_name": ["/tmp/bad_one.png", "good_one.fits"],
            "Problematic Yes/No": ["Yes", "No"],
        }
    ).to_csv(input_path, index=False)

    assert load_problematic_training_thumbs(input_path) == frozenset({"bad_one"})


def test_score_classifications_respects_vote_cap_and_all_same_shortcut() -> None:
    frame = pd.DataFrame(
        {
            "row_order": range(3),
            "classification_id": range(3),
            "user_key": ["u1", "u2", "u3"],
            "user_name": ["u1", "u2", "u3"],
            "thumbname": ["target", "target", "target"],
            "answer": [1.0, 1.0, 0.0],
            "train": [False, False, False],
            "batch": [1, 1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )

    result = score_classifications(
        frame,
        ScoringConfig(max_votes_per_thumb=2, dedupe_user_thumbnail=False, score_all_same_shortcut=True),
    )
    target = result.scores.set_index("thumbname").loc["target"]

    assert target["vote_count"] == 2
    assert target["score"] == 1.0


def test_score_classifications_adds_raw_thumbnail_summary_columns() -> None:
    frame = pd.DataFrame(
        {
            "row_order": range(4),
            "classification_id": range(4),
            "user_key": ["u1", "u1", "u2", "u3"],
            "user_name": ["u1", "u1", "u2", "u3"],
            "thumbname": ["target", "target", "target", "target"],
            "answer": [1.0, 0.0, np.nan, 1.0],
            "train": [False, False, False, False],
            "class": ["object-a", "object-a", "object-a", "object-a"],
            "batch": [1, 1, 1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )

    result = score_classifications(frame, ScoringConfig(dedupe_user_thumbnail=True))
    target = result.scores.set_index("thumbname").loc["target"]

    assert target["vote_count"] == 2
    assert target["number_of_times_classified"] == 4
    assert target["number_of_yes_answers"] == 2
    assert target["number_of_no_answers"] == 1
    assert np.isclose(target["fraction_yes"], 2 / 3)
    assert target["train"] is False or target["train"] == np.False_
    assert target["class"] == "object-a"


def test_time_and_legacy_modes_return_row_weights() -> None:
    frame = pd.DataFrame(
        {
            "row_order": range(6),
            "classification_id": range(6),
            "user_key": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "user_name": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "thumbname": ["train", "a", "b", "train", "a", "b"],
            "answer": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "train": [True, False, False, True, False, False],
            "batch": [1, 1, 1, 1, 1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )

    for mode in ("time", "legacy"):
        row_weights, user_weights = row_weights_for_mode(frame, ScoringConfig(mode=mode))
        assert len(row_weights) == len(frame)
        assert np.isfinite(row_weights).all()
        assert set(user_weights["user_key"]) == {"u1", "u2"}


def test_write_and_compare_score_outputs_average_duplicate_reference(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "row_order": [0, 1],
            "classification_id": [0, 1],
            "user_key": ["u1", "u2"],
            "user_name": ["u1", "u2"],
            "thumbname": ["a", "b"],
            "answer": [1.0, 0.0],
            "train": [False, False],
            "batch": [1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )
    result = score_classifications(frame, ScoringConfig())
    paths = write_score_outputs(result, tmp_path / "out", "candidate", cutoff=0.5, coc_scores_path=None)
    reference_path = tmp_path / "reference.csv"
    diff_path = tmp_path / "diff.csv"
    pd.DataFrame(
        {
            "thumb_name": ["a.png", "a.png", "b.png"],
            "score": [1.0, 1.0, 0.0],
        }
    ).to_csv(reference_path, index=False)

    report = compare_score_files(reference_path, paths["total"], diff_path)
    total = pd.read_csv(paths["total"])
    filtered = pd.read_csv(paths["filtered"])

    assert report["reference_rows"] == 3
    assert report["reference_duplicate_rows"] == 2
    assert report["matched_thumbnails"] == 2
    assert diff_path.exists()
    assert total.columns.tolist() == SCORE_OUTPUT_COLUMNS
    assert filtered.columns.tolist() == SCORE_OUTPUT_COLUMNS


@pytest.mark.parametrize(
    ("coc_text", "expected"),
    [
        ("thumbnail_basename,score\nactive.png,8.0\nquiet.png,2.0\ncoc_only.png,9.0\n", 8.0),
        ("thumbname,score\nactive.png,7.0\nquiet.png,2.0\ncoc_only.png,9.0\n", 7.0),
        ("path,score\n/tmp/active.png,6.0\n/tmp/quiet.png,2.0\n/tmp/coc_only.png,9.0\n", 6.0),
    ],
)
def test_write_score_outputs_adds_coc_score_to_total_and_filtered(tmp_path, coc_text: str, expected: float) -> None:
    frame = pd.DataFrame(
        {
            "row_order": [0, 1],
            "classification_id": [1, 2],
            "user_key": ["u1", "u2"],
            "user_name": ["u1", "u2"],
            "thumbname": ["active", "quiet"],
            "answer": [1.0, 0.0],
            "train": [False, False],
            "batch": [1, 1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )
    coc_path = tmp_path / "coc_scores.csv"
    coc_path.write_text(coc_text)

    result = score_classifications(frame, ScoringConfig(score_all_same_shortcut=True))
    paths = write_score_outputs(
        result,
        tmp_path / "out",
        "coc",
        cutoff=0.5,
        coc_scores_path=coc_path,
        require_coc_scores=True,
    )
    total = pd.read_csv(paths["total"])
    filtered = pd.read_csv(paths["filtered"])

    assert total.columns.tolist() == SCORE_OUTPUT_COLUMNS
    assert filtered.columns.tolist() == SCORE_OUTPUT_COLUMNS
    assert total["thumbname"].tolist() == ["active", "quiet"]
    assert "coc_only" not in set(total["thumbname"])
    assert total.set_index("thumbname").loc["active", "coc_score"] == expected
    assert total.set_index("thumbname").loc["quiet", "coc_score"] == 2.0
    assert filtered["thumbname"].tolist() == ["active"]
    assert filtered.loc[0, "coc_score"] == expected
    assert result.diagnostics["coc_scores_status"] == "loaded"
    assert result.diagnostics["coc_matched_rows"] == 2


def test_write_score_outputs_keeps_last_duplicate_coc_score(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "row_order": [0],
            "classification_id": [1],
            "user_key": ["u1"],
            "user_name": ["u1"],
            "thumbname": ["dupe"],
            "answer": [1.0],
            "train": [False],
            "batch": [1],
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )
    coc_path = tmp_path / "coc_scores.csv"
    coc_path.write_text("thumbnail_basename,score\ndupe.png,1\ndupe.png,4\n")

    result = score_classifications(frame, ScoringConfig())
    paths = write_score_outputs(result, tmp_path / "out", "dupe", 0.5, coc_scores_path=coc_path)
    total = pd.read_csv(paths["total"])

    assert total.loc[0, "coc_score"] == 4.0
    assert result.diagnostics["coc_duplicate_rows_removed"] == 1


def test_cli_scores_active_asteroids_csv_with_progress(tmp_path, capsys) -> None:
    input_path = tmp_path / "classifications.csv"
    output_dir = tmp_path / "results"
    pd.DataFrame(
        {
            "classification_id": [1, 2],
            "username": ["u1", "u2"],
            "thumbname": ["active.png", "quiet.png"],
            "value": ["yes", "no"],
            "train": [None, None],
            "class": ["object-active", "object-quiet"],
            "batch": [1, 1],
        }
    ).to_csv(input_path, index=False)

    status = main(
        [
            "score",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--label",
            "active",
            "--mode",
            "naive",
            "--progress",
            "always",
            "--no-coc-scores",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "Loading input table..." in captured.err
    assert '"diagnostics"' in captured.out
    total = pd.read_csv(output_dir / "output_total_active.csv", keep_default_na=False)
    filtered = pd.read_csv(output_dir / "output_active.csv", keep_default_na=False)
    assert total.columns.tolist() == SCORE_OUTPUT_COLUMNS
    assert filtered.columns.tolist() == SCORE_OUTPUT_COLUMNS
    assert total.set_index("thumbname").loc["active", "coc_score"] == ""
    assert total.set_index("thumbname").loc["active", "number_of_yes_answers"] == 1
    assert total.set_index("thumbname").loc["active", "class"] == "object-active"
    assert (output_dir / "user_weights_active.csv").exists()


def test_cli_scores_active_asteroids_with_explicit_coc_scores(tmp_path, capsys) -> None:
    input_path = tmp_path / "classifications.csv"
    output_dir = tmp_path / "results"
    coc_path = tmp_path / "coc_scores.csv"
    pd.DataFrame(
        {
            "classification_id": [1, 2],
            "username": ["u1", "u2"],
            "thumbname": ["active.png", "quiet.png"],
            "value": ["yes", "no"],
            "train": [None, None],
        }
    ).to_csv(input_path, index=False)
    coc_path.write_text("thumbnail_basename,score\nactive.png,5\n")

    status = main(
        [
            "score",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--label",
            "active",
            "--mode",
            "naive",
            "--all-same-shortcut",
            "--coc-scores",
            str(coc_path),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert '"coc_scores_status": "loaded"' in captured.out
    total = pd.read_csv(output_dir / "output_total_active.csv")
    filtered = pd.read_csv(output_dir / "output_active.csv")
    assert total.set_index("thumbname").loc["active", "coc_score"] == 5.0
    assert pd.isna(total.set_index("thumbname").loc["quiet", "coc_score"])
    assert filtered.loc[0, "thumbname"] == "active"
    assert filtered.loc[0, "coc_score"] == 5.0


def test_cli_explicit_missing_coc_scores_path_is_an_error(tmp_path) -> None:
    input_path = tmp_path / "classifications.csv"
    pd.DataFrame(
        {
            "classification_id": [1],
            "username": ["u1"],
            "thumbname": ["active.png"],
            "value": ["yes"],
            "train": [None],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(FileNotFoundError):
        main(["score", str(input_path), "--coc-scores", str(tmp_path / "missing.csv")])


def test_cli_compare_and_inspect_inputs(tmp_path, capsys) -> None:
    old_csv = tmp_path / "old.csv"
    new_parquet = tmp_path / "new.parquet"
    pd.DataFrame(
        {
            "classification_id": [1],
            "username": ["u1"],
            "thumbname": ["same.png"],
            "value": ["yes"],
            "train": ["false"],
        }
    ).to_csv(old_csv, index=False)
    pd.DataFrame(
        {
            "classification_id": [1],
            "user_name": ["u1"],
            "thumbname": ["same.png"],
            "answer": [1.0],
            "train": [False],
        }
    ).to_parquet(new_parquet, index=False)

    assert main(["inspect-inputs", str(old_csv), str(new_parquet)]) == 0
    inspect_output = capsys.readouterr().out
    assert '"old_rows": 1' in inspect_output

    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    pd.DataFrame({"thumbname": ["same"], "score": [1.0]}).to_csv(reference_path, index=False)
    pd.DataFrame({"thumbname": ["same"], "score": [0.8]}).to_csv(candidate_path, index=False)

    assert main(["compare", str(reference_path), str(candidate_path)]) == 0
    compare_output = capsys.readouterr().out
    assert '"matched_thumbnails": 1' in compare_output
