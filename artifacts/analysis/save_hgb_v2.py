"""Refit the selected rich model on the cached K=12 lists and save a new artifact.

Uses the first 70% only. The decision rule is the one already chosen on the
tune slice. The last 30% is scored and not used for fitting.
Does not touch artifacts/rank_model.joblib.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "artifacts/analysis")

import joblib

from decision_stage import FEATURE_NAMES, apply_rule, fit_hgb, matrix, metrics_of, pool_stats, proba_of

RULE = {
    "t": 0.72,
    "t_empty": 0.82,
    "t_extra": 0.72,
    "t_short": 0.82,
    "t_conflict": 0.72,
    "margin": 0.12,
}


def main() -> None:
    rows = pickle.loads(Path("artifacts/analysis/k12_candidates.pkl").read_bytes())
    split = int(len(rows) * 0.7)
    train, holdout = rows[:split], rows[split:]
    print("holdout_candidates", pool_stats(holdout), flush=True)
    X, y, _ = matrix(train)
    model = fit_hgb(X, y)
    preds = apply_rule(holdout, proba_of(model, holdout, "rich"), RULE)
    metrics = metrics_of(holdout, preds)
    print("refit_holdout", metrics, flush=True)
    out = Path("artifacts/rank_model_hgb_v2.joblib")
    joblib.dump(
        {
            "model": model,
            "rule": RULE,
            "feature_names": FEATURE_NAMES,
            "trained_on": "stride 400, first 70%",
        },
        out,
    )
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
