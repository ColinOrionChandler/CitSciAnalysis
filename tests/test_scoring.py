from __future__ import annotations

import numpy as np
import pandas as pd

from citsci_analysis.scoring import (
    ScoringConfig,
    calculate_metric_weight,
    normalize_thumbnames,
    score_classifications,
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
