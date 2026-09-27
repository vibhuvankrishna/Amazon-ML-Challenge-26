"""Build the corrected test indexes, then infer into output_v5.

Does not write output_v4, test_blocks_union.sqlite, or the HGB model.
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
    extra = ROOT / "artifacts/test_blocks_extra_v5.sqlite"
    legacy = ROOT / "artifacts/test_blocks_legacy_v5.sqlite"
    print("PREFLIGHT", flush=True)
    print("test_s1_rows=1732544", flush=True)
    print("expected_output_rows=1732544", flush=True)
    print(f"legacy_index={legacy}", flush=True)
    print(f"extra_index={extra}", flush=True)
    print("rank_model=artifacts/rank_model_hgb_v2.joblib", flush=True)
    print("rule_path=artifacts/rule_v2.json", flush=True)
    print(f"K=12 extra_cap={EXTRA_CAP} legacy_cap={MAX_PER_KEY}", flush=True)
    print("output_dir=output_v5", flush=True)
    if not index_ready(extra):
        build_extra_exact_db(
            [
                ROOT / "dataset/test/test_source2.tsv",
                ROOT / "dataset/test/test_source3.tsv",
            ],
            extra,
            EXTRA_CAP,
        )
    else:
        print(f"Reusing complete extra index {extra}", flush=True)
    if not index_ready(legacy):
        build_legacy_block_db_from_sqlite(
            ROOT / "artifacts/test_s23.sqlite",
            legacy,
            MAX_PER_KEY,
        )
    else:
        print(f"Reusing complete legacy index {legacy}", flush=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "code/business_entity_resolution/src/rank_match.py"),
        "--mode",
        "infer",
        "--decision-model",
        str(ROOT / "artifacts/rank_model_hgb_v2.joblib"),
        "--rule-path",
        str(ROOT / "artifacts/rule_v2.json"),
        "--test-sqlite",
        str(ROOT / "artifacts/test_s23.sqlite"),
        "--legacy-block-db",
        str(legacy),
        "--extra-block-db",
        str(extra),
        "--output-dir",
        str(ROOT / "output_v5"),
        "--max-cands",
        "12",
    ]
    print("INFER", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
