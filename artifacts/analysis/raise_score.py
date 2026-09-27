"""Raise-score pipeline: refit rich HGB with hard negatives, save model, optional infer.

Rebuilds artifacts/rank_model_hgb_v3.joblib from the K=12 candidate cache (or
rebuilds that cache when train block indexes + dataset are present).

Does not overwrite rank_model_hgb_v2.joblib.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
sys.path.insert(0, str(ROOT / "artifacts/analysis"))

from decision_stage import (  # noqa: E402
    CACHE,
    FEATURE_NAMES,
    apply_rule,
    build_cache,
    fit_hgb,
    hard_negative_weights,
    matrix,
    metrics_of,
    pool_stats,
    proba_of,
)

RULE = {
    "t": 0.72,
    "t_empty": 0.82,
    "t_extra": 0.72,
    "t_short": 0.82,
    "t_conflict": 0.72,
    "margin": 0.12,
}

OUT_MODEL = ROOT / "artifacts/rank_model_hgb_v3.joblib"
REPORT = ROOT / "artifacts/analysis/raise_score_report.json"


def load_or_build_rows(rebuild: bool) -> list[dict]:
    if CACHE.exists() and not rebuild:
        rows = pickle.loads(CACHE.read_bytes())
        print(f"loaded {CACHE} n={len(rows)}", flush=True)
        return rows
    train_db = ROOT / "artifacts/train_blocks_v2.sqlite"
    s1 = ROOT / "dataset/train/train_source1.tsv"
    if not train_db.exists() or not s1.exists():
        raise SystemExit(
            "Need dataset/train/*.tsv and artifacts/train_blocks_v2.sqlite "
            f"(or existing {CACHE}) to rebuild the candidate cache."
        )
    print("Building K=12 candidate cache…", flush=True)
    return build_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--min-f05", type=float, default=0.83)
    ap.add_argument("--infer", action="store_true", help="Run full test infer if holdout clears min-f05")
    args = ap.parse_args()

    rows = load_or_build_rows(args.rebuild_cache)
    split = int(len(rows) * 0.7)
    train, holdout = rows[:split], rows[split:]
    print("holdout_candidates", pool_stats(holdout), flush=True)

    # Fit on first 70%; rule is the previously validated production rule.
    X, y, _ = matrix(train)
    hw = hard_negative_weights(train)
    print(
        f"train_pairs={len(y)} positives={int(y.sum())} "
        f"hard_neg_mean_w={float(hw[y == 0].mean()) if (y == 0).any() else 0:.2f}",
        flush=True,
    )
    model = fit_hgb(X, y, sample_weight=hw)
    preds = apply_rule(holdout, proba_of(model, holdout, "rich"), RULE)
    metrics = metrics_of(holdout, preds)
    print("holdout_metrics", json.dumps(metrics), flush=True)

    blob = {
        "model": model,
        "rule": RULE,
        "feature_names": FEATURE_NAMES,
        "trained_on": "stride 400, first 70%, hard-negative weights",
        "holdout_macro_f05": metrics["macro_f05"],
        "sibling_accept": True,
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(blob, OUT_MODEL)
    REPORT.write_text(json.dumps({"holdout": metrics, "rule": RULE, "model": str(OUT_MODEL)}, indent=2))
    print(f"WROTE {OUT_MODEL}", flush=True)
    print(f"WROTE {REPORT}", flush=True)

    if metrics["macro_f05"] < args.min_f05:
        raise SystemExit(
            f"Holdout macro F0.5 {metrics['macro_f05']:.4f} is below {args.min_f05}. "
            "Not starting full test inference."
        )

    if args.infer:
        from run_output_v5 import main as infer_main

        # Point infer at v3 model via env override by rewriting argv path in a small wrapper.
        print("Holdout cleared threshold; starting test inference…", flush=True)
        import subprocess

        legacy = ROOT / "artifacts/test_blocks_legacy_v5.sqlite"
        extra = ROOT / "artifacts/test_blocks_extra_v5.sqlite"
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "code/business_entity_resolution/src/rank_match.py"),
            "--mode",
            "infer",
            "--decision-model",
            str(OUT_MODEL),
            "--rule-path",
            str(ROOT / "artifacts/rule_v2.json"),
            "--test-sqlite",
            str(ROOT / "artifacts/test_s23.sqlite"),
            "--legacy-block-db",
            str(legacy),
            "--extra-block-db",
            str(extra),
            "--output-dir",
            str(ROOT / "output_v6"),
            "--max-cands",
            "12",
        ]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
