"""Compare synthetic outputs with the pre-migration commit in isolated source trees."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

BASELINE = "cd96e14557b3026556ca10a1767859a62cd92182"


def snapshot(output: Path) -> None:
    """Exercise all modes, repeated votes, COC sidecars, and registry/chunk contracts."""
    from citsci_analysis.rubin import RubinScoringConfig, score_rubin_chunks, write_rubin_outputs
    from citsci_analysis.scoring import (
        ScoringConfig,
        load_classifications,
        score_classifications,
        write_score_outputs,
    )

    output.mkdir()
    # The pair repeats after 7 * 13 = 91 rows, crossing every tested chunk size.
    raw = pd.DataFrame(
        [
            {
                "classification_id": i,
                "user_name": f"synthetic{i % 7}",
                "thumbname": f"image{i % 13}",
                "answer": float(i % 3 != 0),
                "train": i % 13 < 2,
                "subject_id": None if i < 20 else f"subject{i % 13}",
                "object_id": None if i < 30 else f"object{i % 3}",
                "review_unit_id": f"review{i % 13}",
                "batch": None if i < 40 else 1,
                "class": "synthetic",
            }
            for i in range(182)
        ]
    )
    assert raw.duplicated(["user_name", "thumbname"]).sum() == 91
    raw.to_csv(output / "input.csv", index=False)
    raw.to_parquet(output / "input.parquet", index=False)
    coc = output / "coc.csv"
    coc.write_text("thumbnail_basename,score\nimage2,0.1\nimage2,0.7\nimage3,0.3\n")
    for mode in ("naive", "time", "legacy"):
        for fmt in ("csv", "parquet"):
            for dedupe in (True, False):
                label = f"{mode}_{fmt}_{dedupe}"
                result = score_classifications(
                    load_classifications(output / f"input.{fmt}"),
                    ScoringConfig(
                        mode=mode,
                        dedupe_user_thumbnail=dedupe,
                        max_votes_per_thumb=15,
                        score_all_same_shortcut=mode == "legacy",
                    ),
                )
                write_score_outputs(
                    result, output, label, 0.473, coc_scores_path=coc, require_coc_scores=True
                )
                (output / f"{label}.json").write_text(json.dumps(result.diagnostics, sort_keys=True))
    for dedupe in (True, False):
        for chunk_size in (1, 11, 64):
            label = f"rubin_{dedupe}_{chunk_size}"
            result_rubin = score_rubin_chunks(
                [raw.iloc[i : i + chunk_size].copy() for i in range(0, len(raw), chunk_size)],
                RubinScoringConfig(dedupe_user_review_units=dedupe),
            )
            assert result_rubin.diagnostics["scored_votes"] == (91 if dedupe else 182)
            write_rubin_outputs(result_rubin, output, label, 0.473)
            (output / f"{label}.json").write_text(json.dumps(result_rubin.diagnostics, sort_keys=True))


def check_parity() -> None:
    """Run both revisions with identical dependencies and compare every output."""
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="citsci-parity-") as temporary:
        work = Path(temporary)
        archive = work / "baseline.tar"
        subprocess.run(["git", "archive", BASELINE, "-o", str(archive)], cwd=root, check=True)
        baseline = work / "baseline"
        baseline.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(baseline, filter="data")
        for name, source_root in (("before", baseline), ("after", root)):
            env = dict(os.environ, PYTHONPATH=str(source_root / "src"))
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--snapshot", str(work / name)],
                cwd=work,
                env=env,
                check=True,
            )
        before, after = work / "before", work / "after"
        names = sorted(path.name for path in before.iterdir())
        assert names == sorted(path.name for path in after.iterdir()), "Output schemas/files changed"
        for name in names:
            if name.endswith(".parquet"):
                pd.testing.assert_frame_equal(pd.read_parquet(before / name), pd.read_parquet(after / name))
            else:
                old = (before / name).read_text().replace(str(before), "<OUTPUT>")
                new = (after / name).read_text().replace(str(after), "<OUTPUT>")
                assert old == new, f"Scientific output differs: {name}"
        print(f"Scientific parity passed: {len(names)} files against {BASELINE}; 91 duplicate pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
    else:
        check_parity()
