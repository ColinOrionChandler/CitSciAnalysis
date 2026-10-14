"""Deterministic workloads: no research files, network, or persistent products."""

from __future__ import annotations

import numpy as np
import pandas as pd

from citsci_analysis.rubin import RubinScoringConfig, _update_registry, score_rubin_chunks
from citsci_analysis.scoring import ScoringConfig, score_classifications


def classifications(size: int) -> pd.DataFrame:
    """Create normalized votes with repeated users, thumbnails, and training data."""
    i = np.arange(size)
    return pd.DataFrame(
        {
            "row_order": i,
            "classification_id": i,
            "user_key": "u" + pd.Series(i % 100).astype(str),
            "user_name": "u" + pd.Series(i % 100).astype(str),
            "thumbname": "image" + pd.Series(i % 501).astype(str),
            "answer": (i % 3 != 0).astype(float),
            "train": i % 7 == 0,
            "class": "synthetic",
            "batch": 1,
            "started_at": pd.NaT,
            "finished_at": pd.NaT,
        }
    )


class Scoring:
    """Classic and streaming scorer runtime and peak resident memory."""

    def setup(self) -> None:
        """Construct reusable inputs outside the timed region."""
        self.frame = classifications(10_000)
        self.raw = self.frame.rename(columns={"user_key": "unused"}).copy()
        self.raw["review_unit_id"] = self.raw["thumbname"]
        self.raw["object_id"] = "object" + pd.Series(np.arange(len(self.raw)) % 20).astype(str)
        # Repeat votes after the original chunks so streaming deduplication is exercised.
        self.raw = pd.concat([self.raw, self.raw.iloc[:2000]], ignore_index=True)
        self.chunks = [self.raw.iloc[start : start + 2000].copy() for start in range(0, len(self.raw), 2000)]
        assert self.raw.duplicated(["user_name", "review_unit_id"]).sum() == 2000

    def time_classic(self) -> None:
        """Time static-weight classic scoring."""
        score_classifications(self.frame, ScoringConfig())

    def peakmem_classic(self) -> None:
        """Measure classic scorer peak memory."""
        self.time_classic()

    def time_rubin(self) -> None:
        """Time chunked Rubin aggregation including duplicate votes."""
        score_rubin_chunks(self.chunks, RubinScoringConfig())

    def peakmem_rubin(self) -> None:
        """Measure Rubin scorer peak memory."""
        self.time_rubin()


class Registry:
    """Registry metadata merging across repeated review units."""

    def setup(self) -> None:
        """Include independent missing metadata fields in the workload."""
        i = np.arange(50_000)
        self.chunk = pd.DataFrame(
            {
                "review_unit_id": "review" + pd.Series(i % 5000).astype(str),
                "object_id": pd.Series(np.where(i % 2 == 0, "synthetic", None), dtype="string"),
                "subject_id": pd.Series(np.where(i % 3 == 0, "subject", None), dtype="string"),
                "batch": np.where(i % 5 == 0, 1.0, np.nan),
            }
        )

    def time_registry(self) -> None:
        """Time first-non-null metadata aggregation."""
        registry: dict[str, dict[str, object]] = {}
        _update_registry(registry, self.chunk)

    def peakmem_registry(self) -> None:
        """Measure registry aggregation peak memory."""
        self.time_registry()
