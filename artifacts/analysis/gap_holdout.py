"""Leakage-safe counterfactuals on the cached K=12 holdout. No test GT. No submission writes."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path("/Users/m.harshithakrishna/Desktop/student_resource")
sys.path.insert(0, str(ROOT / "artifacts/analysis"))
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
csv.field_size_limit(10_000_000)

from ber.metrics import score_entity
from decision_stage import FEATURE_NAMES, apply_rule, proba_of

RULE = {
    "t": 0.72,
    "t_empty": 0.82,
    "t_extra": 0.72,
    "t_short": 0.82,
    "t_conflict": 0.72,
    "margin": 0.12,
}


def macro(rows, preds):
    return float(np.mean([score_entity(p, r["true"]) for r, p in zip(rows, preds)]))


def oracle_preds(rows):
    return [r["true"] & {c["cid"] for c in r["cands"]} for r in rows]


def cut_k(rows, probas, k):
    new_rows, new_ps = [], []
    for row, ps in zip(rows, probas):
        keep = [(c, p) for c, p in zip(row["cands"], ps) if float(c["x"][25]) <= k]
        new_rows.append({"true": row["true"], "cands": [c for c, _ in keep]})
        new_ps.append([p for _, p in keep])
    return new_rows, new_ps


def gate_singleton(rows, probas, rule, min_blockers, min_p):
    preds = apply_rule(rows, probas, rule)
    out = []
    for row, ps, pred in zip(rows, probas, preds):
        if not ps:
            out.append(pred)
            continue
        i = max(range(len(ps)), key=lambda j: ps[j])
        if float(row["cands"][i]["x"][24]) < min_blockers and ps[i] < min_p:
            out.append(set())
        else:
            out.append(pred)
    return out


def micro(rows, preds):
    tp = fp = fn = 0
    for row, pred in zip(rows, preds):
        true = row["true"]
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4)}


def pack(rows, probas, rule):
    preds = apply_rule(rows, probas, rule)
    return {
        "macro_f05": round(macro(rows, preds), 4),
        **micro(rows, preds),
        "empty_pred": round(sum(1 for p in preds if not p) / max(len(preds), 1), 4),
        "avg_pred": round(sum(len(p) for p in preds) / max(len(preds), 1), 4),
    }


def grid(rows, probas):
    best = None
    tried = 0
    for t in (0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88):
        for t_empty in (t, min(0.94, t + 0.10), min(0.96, t + 0.18)):
            for margin in (0.08, 0.12, 0.25):
                for t_short in (t, min(0.94, t + 0.10)):
                    rule = {
                        "t": t,
                        "t_empty": t_empty,
                        "t_extra": t,
                        "t_short": t_short,
                        "t_conflict": t,
                        "margin": margin,
                    }
                    score = macro(rows, apply_rule(rows, probas, rule))
                    tried += 1
                    if best is None or score > best[0]:
                        best = (score, rule)
    return best, tried


def country_map(sids):
    found = {}
    path = ROOT / "dataset/train/train_source1.tsv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            if eid in sids and eid not in found:
                found[eid] = (row.get("country") or "").strip().upper() or "UNK"
                if len(found) == len(sids):
                    break
    return found


def clear_random(rows, preds, n_clear, seed):
    rng = np.random.default_rng(seed)
    idx = [i for i, r in enumerate(rows) if r["cands"]]
    choose = set(rng.choice(idx, size=min(n_clear, len(idx)), replace=False).tolist())
    out = []
    for i, pred in enumerate(preds):
        out.append(set() if i in choose else pred)
    return out


def main():
    rows = pickle.loads((ROOT / "artifacts/analysis/k12_candidates.pkl").read_bytes())
    split = int(len(rows) * 0.7)
    train, hold = rows[:split], rows[split:]
    tune_cut = int(len(train) * 0.75)
    tune = train[tune_cut:]
    blob = joblib.load(ROOT / "artifacts/rank_model_hgb_v2.joblib")
    model = blob["model"]
    saved_rule = blob["rule"]
    print("scoring", flush=True)
    p_hold = proba_of(model, hold, "rich")
    p_tune = proba_of(model, tune, "rich")
    current = pack(hold, p_hold, saved_rule)
    ora = oracle_preds(hold)
    oracle_f = macro(hold, ora)
    base_preds = apply_rule(hold, p_hold, saved_rule)

    # Gap attribution. Primary bucket is exclusive.
    buckets = Counter()
    gap_oracle = Counter()
    gap_perfect = Counter()
    for row, pred in zip(hold, base_preds):
        true = row["true"]
        cand = {c["cid"] for c in row["cands"]}
        ora_set = true & cand
        sm = score_entity(pred, true)
        so = score_entity(ora_set, true)
        missed_in = ora_set - pred
        fp = pred - true
        outside = true - cand
        if not true and not pred:
            key = "correct_singleton"
        elif not true and pred:
            key = "false_singleton"
        elif not outside and not missed_in and not fp:
            key = "exact"
        elif outside and not ora_set:
            key = "all_true_outside_list"
        elif missed_in and not fp:
            key = "rejected_true_in_list"
        elif fp and not missed_in:
            key = "false_added"
        else:
            key = "mixed_miss_and_false"
        buckets[key] += 1
        gap_oracle[key] += so - sm
        gap_perfect[key] += 1.0 - sm

    (best_tune, best_rule), tried = grid(tune, p_tune)
    tuned = pack(hold, p_hold, best_rule)

    declared = {
        "current": saved_rule,
        "conservative": {
            "t": 0.80,
            "t_empty": 0.90,
            "t_extra": 0.84,
            "t_short": 0.90,
            "t_conflict": 0.84,
            "margin": 0.08,
        },
        "aggressive": {
            "t": 0.60,
            "t_empty": 0.72,
            "t_extra": 0.60,
            "t_short": 0.70,
            "t_conflict": 0.60,
            "margin": 0.20,
        },
    }
    declared_scores = {name: pack(hold, p_hold, rule) for name, rule in declared.items()}

    # Upper bound: best rule in the same grid scored directly on holdout.
    (best_hold_f, best_hold_rule), _ = grid(hold, p_hold)

    k_scores = {}
    for k in (4, 8, 12):
        rr, pp = cut_k(hold, p_hold, k)
        k_scores[str(k)] = {
            "model": pack(rr, pp, saved_rule),
            "oracle": round(macro(rr, oracle_preds(rr)), 4),
            "avg_cand": round(sum(len(r["cands"]) for r in rr) / len(rr), 3),
        }

    gates = {
        "drop_if_best_blocker_lt2_and_p_lt_0.85": gate_singleton(hold, p_hold, saved_rule, 2, 0.85),
        "drop_if_best_blocker_lt2_and_p_lt_0.78": gate_singleton(hold, p_hold, saved_rule, 2, 0.78),
    }
    gate_scores = {}
    for name, preds in gates.items():
        gate_scores[name] = {
            "macro_f05": round(macro(hold, preds), 4),
            **micro(hold, preds),
            "empty_pred": round(sum(1 for p in preds if not p) / len(preds), 4),
        }

    # Country-conditional threshold, selected on tune only.
    countries = country_map({r["sid"] for r in train + hold})
    by_tune = defaultdict(list)
    for row, ps in zip(tune, p_tune):
        by_tune[countries.get(row["sid"], "UNK")].append((row, ps))
    local_t = {}
    for cc, pairs in by_tune.items():
        if len(pairs) < 40:
            continue
        sub_rows = [a for a, _ in pairs]
        sub_ps = [b for _, b in pairs]
        best_t, best_s = 0.72, -1.0
        for t in (0.60, 0.68, 0.72, 0.78, 0.84):
            rule = dict(saved_rule)
            rule["t"] = t
            rule["t_extra"] = t
            rule["t_conflict"] = t
            s = macro(sub_rows, apply_rule(sub_rows, sub_ps, rule))
            if s > best_s:
                best_t, best_s = t, s
        local_t[cc] = best_t

    cond_preds = []
    for row, ps in zip(hold, p_hold):
        rule = dict(saved_rule)
        t = local_t.get(countries.get(row["sid"], "UNK"))
        if t is not None:
            rule["t"] = t
            rule["t_extra"] = t
            rule["t_conflict"] = t
        cond_preds.append(apply_rule([row], [ps], rule)[0])
    conditional = {
        "macro_f05": round(macro(hold, cond_preds), 4),
        **micro(hold, cond_preds),
        "countries_with_own_threshold": len(local_t),
        "thresholds": local_t,
    }

    # Per-country current score on holdout.
    per_c = defaultdict(list)
    for row, pred in zip(hold, base_preds):
        per_c[countries.get(row["sid"], "UNK")].append(score_entity(pred, row["true"]))
    per_country = [
        {"country": cc, "n": len(xs), "macro_f05": round(float(np.mean(xs)), 4)}
        for cc, xs in per_c.items()
        if len(xs) >= 15
    ]
    per_country.sort(key=lambda r: -r["n"])

    # Feature means on holdout candidates.
    X = np.vstack([c["x"] for r in hold for c in r["cands"]])
    accepted = []
    best = []
    for row, ps, pred in zip(hold, p_hold, base_preds):
        if ps:
            i = int(np.argmax(ps))
            best.append(row["cands"][i]["x"])
        for c in row["cands"]:
            if c["cid"] in pred:
                accepted.append(c["x"])
    A = np.vstack(accepted) if accepted else np.zeros((1, X.shape[1]))
    B = np.vstack(best) if best else np.zeros((1, X.shape[1]))
    feat = []
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        feat.append(
            {
                "feature": name,
                "cand_mean": round(float(col.mean()), 4),
                "cand_p50": round(float(np.median(col)), 4),
                "accepted_mean": round(float(A[:, i].mean()), 4),
                "best_mean": round(float(B[:, i].mean()), 4),
            }
        )
    max_p = np.asarray([max(ps) if ps else 0.0 for ps in p_hold])
    all_p = np.asarray([p for ps in p_hold for p in ps])

    def pct(arr):
        if len(arr) == 0:
            return {}
        return {
            "n": int(len(arr)),
            "min": round(float(arr.min()), 4),
            "p50": round(float(np.median(arr)), 4),
            "mean": round(float(arr.mean()), 4),
            "p90": round(float(np.quantile(arr, 0.90)), 4),
            "p95": round(float(np.quantile(arr, 0.95)), 4),
            "max": round(float(arr.max()), 4),
        }

    n = len(hold)
    zero = sum(1 for r in hold if not r["cands"])
    target_zero = int(round(0.0584 * n))
    n_clear = max(0, target_zero - zero)
    cleared = clear_random(hold, base_preds, n_clear, 0)
    # Also empty the lowest max-proba predictions until empty-pred rate is 15.08%.
    target_empty = int(round(0.1508 * n))
    order = sorted(range(n), key=lambda i: max_p[i])
    forced = [set(p) for p in base_preds]
    empties = sum(1 for p in forced if not p)
    for i in order:
        if empties >= target_empty:
            break
        if forced[i]:
            forced[i] = set()
            empties += 1

    cand_counts = [len(r["cands"]) for r in hold]
    ch = Counter(cand_counts)
    payload = {
        "n_sample": len(rows),
        "n_holdout": n,
        "n_tune": len(tune),
        "saved_rule": saved_rule,
        "current": current,
        "oracle_f05": round(oracle_f, 4),
        "buckets": dict(buckets),
        "oracle_gap_sum": {k: round(v, 3) for k, v in gap_oracle.items()},
        "perfect_gap_sum": {k: round(v, 3) for k, v in gap_perfect.items()},
        "oracle_gap_macro": {k: round(v / n, 4) for k, v in gap_oracle.items()},
        "perfect_gap_macro": {k: round(v / n, 4) for k, v in gap_perfect.items()},
        "grid_tried_on_tune": tried,
        "tune_best_macro": round(best_tune, 4),
        "tune_best_rule": best_rule,
        "tune_best_on_holdout": tuned,
        "holdout_grid_upper_bound": round(best_hold_f, 4),
        "holdout_grid_upper_rule": best_hold_rule,
        "declared": declared_scores,
        "by_k": k_scores,
        "singleton_gates": gate_scores,
        "country_conditional": {
            "macro_f05": conditional["macro_f05"],
            "precision": conditional["precision"],
            "recall": conditional["recall"],
            "fp": conditional["fp"],
            "fn": conditional["fn"],
            "n_countries": conditional["countries_with_own_threshold"],
        },
        "per_country": per_country,
        "features": feat,
        "max_proba": pct(max_p),
        "all_proba": pct(all_p),
        "cand_hist": {str(k): v for k, v in sorted(ch.items())},
        "sim_match_test_zero_cand_rate": {
            "cleared": n_clear,
            "macro_f05": round(macro(hold, cleared), 4),
            "empty_pred": round(sum(1 for p in cleared if not p) / n, 4),
        },
        "sim_match_test_empty_pred_rate": {
            "macro_f05": round(macro(hold, forced), 4),
            "empty_pred": round(sum(1 for p in forced if not p) / n, 4),
        },
    }
    path = ROOT / "artifacts/analysis/gap_holdout.json"
    path.write_text(json.dumps(payload))
    print("current", current["macro_f05"], "oracle", round(oracle_f, 4), flush=True)
    print("tune_best_on_holdout", tuned["macro_f05"], "grid_upper", round(best_hold_f, 4), flush=True)
    print("conservative", declared_scores["conservative"]["macro_f05"], "aggressive", declared_scores["aggressive"]["macro_f05"], flush=True)
    print("sim_zero", payload["sim_match_test_zero_cand_rate"], flush=True)
    print("sim_empty", payload["sim_match_test_empty_pred_rate"], flush=True)
    print("buckets", dict(buckets), flush=True)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
