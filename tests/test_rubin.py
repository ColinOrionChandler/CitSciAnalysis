from __future__ import annotations

import numpy as np
import pandas as pd

from citsci_analysis.cli import main
from citsci_analysis.rubin import RubinScoringConfig, score_rubin_chunks, _update_registry


def test_registry_fills_columns_independently_and_preserves_first_values() -> None:
    registry = {}
    _update_registry(registry, pd.DataFrame({
        "review_unit_id": ["a", "a", "b", None],
        "object_id": [None, "object-a", None, "ignored"],
        "subject_id": ["subject-a", None, "subject-b", "ignored"],
        "batch": [None, 2, None, 99],
    }))
    _update_registry(registry, pd.DataFrame({
        "review_unit_id": ["a", "b"],
        "object_id": ["conflicting-object", "object-b"],
        "subject_id": ["conflicting-subject", None],
        "batch": [3, 4],
    }))
    assert registry == {
        "a": {"review_unit_id": "a", "object_id": "object-a", "subject_id": "subject-a", "batch": 2},
        "b": {"review_unit_id": "b", "object_id": "object-b", "subject_id": "subject-b", "batch": 4},
    }


def _rubin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "classification_id": range(9),
            "user_name": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u1"],
            "thumbname": [
                "train_a.png",
                "obj_a_1.png",
                "obj_a_1.png",
                "train_a.png",
                "obj_a_1.png",
                "obj_a_2.png",
                "obj_b_1.png",
                "obj_b_2.png",
                "obj_a_1.png",
            ],
            "answer": [1, 1, 0, 1, 0, 1, 0, 1, 1],
            "train": [True, False, False, True, False, False, False, False, False],
            "batch": [1] * 9,
            "class": ["train", "object-a", "object-a", "train", "object-a", "object-a", None, None, "object-a"],
            "subject_id": ["train-subject", "sub-a", "sub-a", "train-subject", "sub-a", "sub-a", "subject-b", "subject-b", "sub-a"],
        }
    )


def test_rubin_scoring_dedupes_across_chunks_and_rolls_up_objects() -> None:
    frame = _rubin_frame()
    chunks = [frame.iloc[:4].copy(), frame.iloc[4:].copy()]
    config = RubinScoringConfig(metric_weights=(0.0, 1.0, 0.0), chunk_size=4)

    result = score_rubin_chunks(chunks, config)
    review_units = result.review_unit_scores.set_index("review_unit_id")
    objects = result.object_scores.set_index("object_id")

    assert review_units.loc["obj_a_1", "vote_count"] == 2
    assert review_units.loc["obj_a_1", "yes_count"] == 1
    assert 0.0 < review_units.loc["obj_a_1", "score"] < 1.0
    assert review_units.loc["obj_a_2", "score"] == 1.0

    assert objects.loc["object-a", "best_review_unit"] == "obj_a_2"
    assert np.isclose(
        objects.loc["object-a", "top_3_mean_score"],
        review_units.loc[["obj_a_1", "obj_a_2"], "score"].mean(),
    )
    assert objects.loc["subject-b", "review_unit_count"] == 2
    assert objects.loc["subject-b", "vote_count"] == 2
    assert result.diagnostics["scored_votes"] == 7


def test_rubin_object_id_falls_back_to_filename_prefix() -> None:
    frame = pd.DataFrame(
        {
            "classification_id": [1],
            "user_name": ["u1"],
            "filename": ["2003_QT30_20160309-06430_1188253_reticle.png"],
            "answer": [1],
        }
    )

    result = score_rubin_chunks([frame], RubinScoringConfig())

    assert result.review_unit_scores.loc[0, "review_unit_id"] == "2003_QT30_20160309-06430_1188253_reticle"
    assert result.object_scores.loc[0, "object_id"] == "2003_QT30"


def test_rubin_ids_can_come_from_subject_metadata() -> None:
    frame = pd.DataFrame(
        {
            "classification_id": [1],
            "user_name": ["u1"],
            "thumbname": ["image_fallback.png"],
            "subject_id": ["subject-fallback"],
            "metadata": [{"review_unit_id": "pair-001", "object_id": "comet-001"}],
            "answer": [1],
        }
    )

    result = score_rubin_chunks([frame], RubinScoringConfig())

    assert result.review_unit_scores.loc[0, "review_unit_id"] == "pair-001"
    assert result.object_scores.loc[0, "object_id"] == "comet-001"


def test_rubin_cli_writes_rubin_outputs(tmp_path) -> None:
    input_path = tmp_path / "rubin.csv"
    output_dir = tmp_path / "results"
    _rubin_frame().to_csv(input_path, index=False)

    status = main(
        [
            "score",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--label",
            "test",
            "--mode",
            "rubin",
            "--chunk-size",
            "3",
        ]
    )

    assert status == 0
    review_path = output_dir / "rubin_review_unit_scores_test.csv"
    object_path = output_dir / "rubin_object_scores_test.csv"
    weights_path = output_dir / "rubin_user_weights_test.csv"
    assert review_path.exists()
    assert object_path.exists()
    assert weights_path.exists()

    objects = pd.read_csv(object_path)
    assert {"object_id", "object_score", "best_review_unit"}.issubset(objects.columns)
    assert "object-a" in set(objects["object_id"])


def test_rubin_cli_progress_reports_chunk_passes(tmp_path, capsys) -> None:
    input_path = tmp_path / "rubin.csv"
    output_dir = tmp_path / "results"
    _rubin_frame().to_csv(input_path, index=False)

    status = main(
        [
            "score",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--label",
            "test",
            "--mode",
            "rubin",
            "--chunk-size",
            "3",
            "--progress",
            "always",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "Pass 1/2: user weights" in captured.err
    assert "Pass 2/2: score review units" in captured.err
    assert "Aggregating review unit votes..." in captured.err
    assert "Writing Rubin outputs..." in captured.err
    assert '"diagnostics"' in captured.out
