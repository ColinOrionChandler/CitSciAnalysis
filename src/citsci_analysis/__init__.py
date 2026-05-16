"""Modern scoring tools for Active Asteroids and Rubin Comet Catchers."""

from .consolidate import ConsolidationResult, consolidate_score_directory, discover_score_csvs
from .scoring import (
    DEFAULT_COC_SCORES_PATH,
    ScoringConfig,
    compare_score_files,
    inspect_inputs,
    load_classifications,
    score_classifications,
    write_score_outputs,
)
from .rubin import RubinScoringConfig, score_rubin_chunks, score_rubin_file, write_rubin_outputs

__all__ = [
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
]
