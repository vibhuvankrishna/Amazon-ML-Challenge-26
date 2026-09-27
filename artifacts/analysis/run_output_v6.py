"""Build corrected test indexes (if needed), then infer into output_v6 with HGB v3.

Uses sibling-aware decide_matches in rank_match.py and the recovered
artifacts/rank_model_hgb_v3.joblib. Does not overwrite output_v4/v5 or v2 model.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ["TMPDIR"] = str(ROOT / "artifacts")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from rank_match import (  # noqa: E402
    EXTRA_CAP,
    MAX_PER_KEY,
    build_extra_exact_db,
    build_legacy_block_db_from_sqlite,
)


def index_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='ix_k'"
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def main() -> None:
    extra = ROOT / "artifacts/test_blocks_extra_v6.sqlite"
    legacy = ROOT / "artifacts/test_blocks_legacy_v6.sqlite"
    model = ROOT / "artifacts/rank_model_hgb_v3.joblib"
    test_sqlite = ROOT / "artifacts/test_s23.sqlite"
    s2 = ROOT / "dataset/test/test_source2.tsv"
    s3 = ROOT / "dataset/test/test_source3.tsv"
    s1 = ROOT / "dataset/test/test_source1.tsv"

    print("PREFLIGHT", flush=True)
    for p in (model, s1, s2, s3, test_sqlite):
        print(f"  {'OK' if p.exists() else 'MISSING'} {p}", flush=True)
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Place the challenge dataset under dataset/ "
                "and ensure artifacts/test_s23.sqlite exists (or build it with infer.py)."
            )

    print(f"legacy_index={legacy}", flush=True)
    print(f"extra_index={extra}", flush=True)
    print(f"rank_model={model}", flush=True)
    print("output_dir=output_v6", flush=True)
    print(f"K=12 extra_cap={EXTRA_CAP} legacy_cap={MAX_PER_KEY}", flush=True)

    if not index_ready(extra):
        build_extra_exact_db([s2, s3], extra, EXTRA_CAP)
    else:
        print(f"Reusing complete extra index {extra}", flush=True)
    if not index_ready(legacy):
        build_legacy_block_db_from_sqlite(test_sqlite, legacy, MAX_PER_KEY)
    else:
        print(f"Reusing complete legacy index {legacy}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "code/business_entity_resolution/src/rank_match.py"),
        "--mode",
        "infer",
        "--decision-model",
        str(model),
        "--rule-path",
        str(ROOT / "artifacts/rule_v2.json"),
        "--test-sqlite",
        str(test_sqlite),
        "--legacy-block-db",
        str(legacy),
        "--extra-block-db",
        str(extra),
        "--output-dir",
        str(ROOT / "output_v6"),
        "--max-cands",
        "12",
    ]
    print("INFER", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
