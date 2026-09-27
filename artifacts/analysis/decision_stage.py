"""Decision-stage experiments on the frozen production K=12 lists.

Does not change blockers, K, the frozen joblib, or any submission file.
The last 30% of the stride-400 sample is scored once and is not used to fit
or to choose a threshold.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "code/business_entity_resolution/src")

from ber.metrics import f05, score_entity
from ber.normalize import (
    address_tokens,
    name_tokens,
    normalize_address,
    normalize_name,
    significant_name_tokens,
)
from rank_match import (
    _house_ltr,
    _load_gt_sample,
    _loc_ltr,
    _pin_ltr,
    _scan_ids,
    cut_ranked,
    merge_extra_postings,
    pair_signals,
    pools_from_index,
)

CACHE = Path("artifacts/analysis/k12_candidates.pkl")
REPORT = Path("artifacts/analysis/decision_report.json")
FEATURE_NAMES = [
    "name_s",
    "name_jaccard",
    "name_contain",
    "n_shared_name",
    "rare_shared",
    "name_ratio",
    "name_set",
    "name_lev",
    "exact_name",
    "name_len_ratio",
    "short_shared",
    "short_query",
    "addr_sort",
    "addr_set",
    "addr_jaccard",
    "addr_ratio",
    "addr_contain",
    "exact_addr",
    "house",
    "pin",
    "loc",
    "house_conflict",
    "pin_conflict",
    "addr_empty",
    "n_blockers",
    "block_pos",
    "name_and_house",
    "name_and_pin",
    "weak_name_strong_addr",
    "name_agree_addr_conflict",
]


def _toks(name: str) -> list[str]:
    return [t for t in (significant_name_tokens(name) or name_tokens(name)) if len(t) >= 2]


def _jacc(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contain(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def featurize(query, cand, n_blockers: int, block_pos: int) -> np.ndarray:
    qn, qa, _qc = query
    cn, ca, _cc = cand
    sig = pair_signals(qn, qa, _qc, cn, ca, _cc)
    if sig is None:
        name_s, addr_sort, addr_set = 0.0, -1.0, -1.0
    else:
        name_s, addr_sort, addr_set = sig
    qt, ct = set(_toks(qn)), set(_toks(cn))
    shared = qt & ct
    nn1, nn2 = normalize_name(qn), normalize_name(cn)
    aa1, aa2 = normalize_address(qa), normalize_address(ca)
    name_ratio = fuzz.ratio(nn1, nn2) / 100.0 if nn1 and nn2 else 0.0
    name_set = fuzz.token_set_ratio(nn1, nn2) / 100.0 if nn1 and nn2 else 0.0
    name_lev = Levenshtein.normalized_similarity(nn1, nn2) if nn1 and nn2 else 0.0
    exact_name = 1.0 if nn1 and nn1 == nn2 else 0.0
    name_len_ratio = min(len(nn1), len(nn2)) / max(len(nn1), len(nn2)) if nn1 and nn2 else 0.0
    at, bt = set(address_tokens(qa or "")), set(address_tokens(ca or ""))
    addr_j = _jacc(at, bt)
    addr_ratio = fuzz.ratio(aa1, aa2) / 100.0 if aa1 and aa2 else 0.0
    exact_addr = 1.0 if aa1 and aa1 == aa2 else 0.0
    hq, hc = _house_ltr(qa), _house_ltr(ca)
    pq, pc = _pin_ltr(qa), _pin_ltr(ca)
    lq, lc = _loc_ltr(qa), _loc_ltr(ca)
    house = 1.0 if hq and hq == hc else 0.0
    pin = 1.0 if pq and pq == pc else 0.0
    loc = 1.0 if lq and lq == lc else 0.0
    house_conflict = 1.0 if hq and hc and hq != hc else 0.0
    pin_conflict = 1.0 if pq and pc and pq != pc else 0.0
    addr_empty = 1.0 if addr_sort < 0 else 0.0
    addr_sort_f = 0.0 if addr_sort < 0 else addr_sort
    addr_set_f = 0.0 if addr_set < 0 else addr_set
    short_query = 1.0 if len(qt) <= 1 or len(nn1) < 8 else 0.0
    name_agree_addr_conflict = 1.0 if name_s >= 0.75 and addr_empty == 0.0 and addr_sort_f < 0.25 and house == 0.0 and pin == 0.0 else 0.0
    vals = [
        name_s,
        _jacc(qt, ct),
        _contain(qt, ct),
        float(len(shared)),
        float(sum(1 for t in shared if len(t) >= 6)),
        name_ratio,
        name_set,
        name_lev,
        exact_name,
        name_len_ratio,
        float(sum(1 for t in shared if len(t) <= 4)),
        short_query,
        addr_sort_f,
        addr_set_f,
        addr_j,
        addr_ratio,
        _contain(at, bt),
        exact_addr,
        house,
        pin,
        loc,
        house_conflict,
        pin_conflict,
        addr_empty,
        float(n_blockers),
        float(block_pos),
        name_s * house,
        name_s * pin,
        (1.0 if name_s < 0.55 else 0.0) * addr_sort_f,
        name_agree_addr_conflict,
    ]
    return np.asarray(vals, dtype=np.float32)


def x4_of(row: np.ndarray) -> np.ndarray:
    """The frozen model's four inputs. name_s is col 0; addr_sort col 12; we recompute empty from col 23."""
    name_s = float(row[0])
    # Recover the raw pair_signals address scores: empty flag is col 23.
    # addr_sort/addr_set were stored as 0 when empty. The frozen model used -1 and a flag.
    empty = float(row[23]) >= 1.0
    addr_sort = -1.0 if empty else float(row[12])
    addr_set = -1.0 if empty else float(row[13])
    return np.asarray([name_s, addr_sort, max(addr_set, -1.0), 1.0 if addr_sort < 0 else 0.0], dtype=np.float32)


def build_cache() -> list[dict]:
    gt = _load_gt_sample("dataset/train/train_ground_truth.tsv", 400)
    s1 = _scan_ids("dataset/train/train_source1.tsv", {sid for sid, _ in gt})
    packed = [(sid, true, s1[sid]) for sid, true in gt if sid in s1]
    records = [(sid, *query) for sid, _, query in packed]
    print(f"sampled_s1={len(packed)}", flush=True)
    conn = sqlite3.connect("artifacts/train_blocks_v2.sqlite")
    pools = pools_from_index(conn, records)
    conn.close()
    texts: dict = {}
    merge_extra_postings(
        records,
        [Path("dataset/train/train_source2.tsv"), Path("dataset/train/train_source3.tsv")],
        pools,
        texts,
    )
    rows = []
    for sid, true, query in packed:
        sources = pools.get(sid, {})
        order = cut_ranked(query, sources, texts, 12)
        cands = []
        for pos, cid in enumerate(order, start=1):
            other = texts.get(cid)
            if other is None:
                continue
            cands.append(
                {
                    "cid": cid,
                    "x": featurize(query, other, len(sources.get(cid, ())), pos),
                    "n_blockers": len(sources.get(cid, ())),
                }
            )
        rows.append({"sid": sid, "true": true, "cands": cands})
    CACHE.write_bytes(pickle.dumps(rows, protocol=4))
    print(f"WROTE {CACHE}", flush=True)
    return rows


def pool_stats(rows) -> dict:
    bt = bf = 0
    oracle = []
    cc = []
    for row in rows:
        cset = {c["cid"] for c in row["cands"]}
        true = row["true"]
        bt += len(true)
        bf += len(true & cset)
        oracle.append(score_entity(true & cset, true))
        cc.append(len(cset))
    return {
        "pair_recall": round(bf / max(bt, 1), 4),
        "retrieved": bf,
        "true_links": bt,
        "entities": len(rows),
        "avg": round(sum(cc) / max(len(cc), 1), 2),
        "oracle_f05": round(sum(oracle) / max(len(oracle), 1), 4),
        "zero_pct": round(sum(1 for n in cc if n == 0) / max(len(cc), 1), 4),
    }


def predict_frozen(rows, model, threshold: float) -> list[set[str]]:
    preds = []
    probas = []
    for row in rows:
        if not row["cands"]:
            preds.append(set())
            probas.append([])
            continue
        X = np.vstack([x4_of(c["x"]) for c in row["cands"]])
        p = model.predict_proba(X)[:, 1]
        pred = {c["cid"] for c, pi in zip(row["cands"], p) if pi >= threshold}
        preds.append(pred)
        probas.append(p.tolist())
    return preds, probas


def metrics_of(rows, preds) -> dict:
    scores = []
    tp = fp = fn = 0
    single_n = single_ok = 0
    multi_n = multi_exact = 0
    for row, pred in zip(rows, preds):
        true = row["true"]
        scores.append(score_entity(pred, true))
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
        if not true:
            single_n += 1
            single_ok += int(not pred)
        if len(true) >= 2:
            multi_n += 1
            multi_exact += int(pred == true)
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    cand = pool_stats(rows)
    return {
        "macro_f05": round(sum(scores) / max(len(scores), 1), 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f05": round(f05(micro_p, micro_r), 4),
        "false_matches": fp,
        "missed_matches": fn,
        "true_positives": tp,
        "singleton_accuracy": round(single_ok / max(single_n, 1), 4),
        "singletons": single_n,
        "singletons_correct": single_ok,
        "multi_exact_accuracy": round(multi_exact / max(multi_n, 1), 4),
        "multi_entities": multi_n,
        "candidate_recall": cand["pair_recall"],
        "oracle_f05": cand["oracle_f05"],
    }


def categorize(rows, preds, probas) -> dict:
    flags = Counter()
    primary = Counter()
    gap = Counter()
    gap_sum = Counter()
    for row, pred, ps in zip(rows, preds, probas):
        true = row["true"]
        cands = [c["cid"] for c in row["cands"]]
        cset = set(cands)
        pmap = {cid: p for cid, p in zip(cands, ps)}
        in_c = true & cset
        missed_by_clf = in_c - pred
        false = pred - true
        outside = true - cset
        a = bool(missed_by_clf)
        b = bool(false) and bool(pred & true)
        false_outranks = False
        if in_c and ps:
            true_ps = [pmap[cid] for cid in in_c if cid in pmap]
            false_ps = [pmap[cid] for cid in cset - true if cid in pmap]
            if true_ps and false_ps and max(false_ps) > min(true_ps):
                false_outranks = True
        c = false_outranks
        d = bool(outside)
        e = (not true) and bool(pred)
        f = len(true) >= 2 and 0 < len(pred & true) < len(true)
        if a:
            flags["A_clf_misses_true_in_list"] += 1
        if b:
            flags["B_true_kept_and_false_added"] += 1
        if c:
            flags["C_false_outranks_true"] += 1
        if d:
            flags["D_true_not_in_top12"] += 1
        if e:
            flags["E_singleton_false_match"] += 1
        if f:
            flags["F_incomplete_multi"] += 1
        if pred == true:
            primary["correct"] += 1
            label = "correct"
        elif e:
            primary["E"] += 1
            label = "E"
        elif d:
            primary["D"] += 1
            label = "D"
        elif false and a:
            primary["B_and_A"] += 1
            label = "B_and_A"
        elif false:
            primary["B"] += 1
            label = "B"
        elif a:
            primary["A"] += 1
            label = "A"
        elif f:
            primary["F"] += 1
            label = "F"
        else:
            primary["G"] += 1
            label = "G"
        oracle = score_entity(in_c, true)
        actual = score_entity(pred, true)
        delta = oracle - actual
        if delta > 1e-9:
            gap[label] += 1
            gap_sum[label] += delta
    n = len(rows)
    return {
        "entities": n,
        "flags": {k: {"n": v, "pct": round(v / n, 4)} for k, v in flags.items()},
        "primary": {k: {"n": v, "pct": round(v / n, 4)} for k, v in primary.items()},
        "oracle_minus_actual_by_primary": {
            k: {"entities": gap[k], "gap_sum": round(gap_sum[k], 4)} for k in gap_sum
        },
    }


def feature_contrast(rows, preds) -> dict:
    tp, fp = [], []
    for row, pred in zip(rows, preds):
        true = row["true"]
        for c in row["cands"]:
            if c["cid"] in true and c["cid"] in pred:
                tp.append(c["x"])
            elif c["cid"] in pred and c["cid"] not in true:
                fp.append(c["x"])
            elif c["cid"] in true and c["cid"] not in pred:
                tp.append(c["x"])  # missed trues still describe the true class
    # Rebuild true/false candidate matrices without mixing missed into the FP contrast.
    true_x, false_x, missed_x = [], [], []
    for row, pred in zip(rows, preds):
        true = row["true"]
        for c in row["cands"]:
            if c["cid"] in true:
                true_x.append(c["x"])
                if c["cid"] not in pred:
                    missed_x.append(c["x"])
            elif c["cid"] in pred:
                false_x.append(c["x"])
    def mean(xs):
        if not xs:
            return None
        return np.mean(np.vstack(xs), axis=0)

    mt, mf, mm = mean(true_x), mean(false_x), mean(missed_x)
    rows_out = []
    for i, name in enumerate(FEATURE_NAMES):
        rows_out.append(
            {
                "feature": name,
                "true_mean": None if mt is None else round(float(mt[i]), 4),
                "false_selected_mean": None if mf is None else round(float(mf[i]), 4),
                "missed_true_mean": None if mm is None else round(float(mm[i]), 4),
                "true_minus_false": None
                if mt is None or mf is None
                else round(float(mt[i] - mf[i]), 4),
            }
        )
    rows_out.sort(key=lambda r: abs(r["true_minus_false"] or 0), reverse=True)
    return {
        "n_true_in_list": len(true_x),
        "n_false_selected": len(false_x),
        "n_missed_true_in_list": len(missed_x),
        "by_feature": rows_out,
    }


def hard_patterns(rows, preds) -> dict:
    patterns = Counter()
    for row, pred in zip(rows, preds):
        true = row["true"]
        for c in row["cands"]:
            if c["cid"] not in pred or c["cid"] in true:
                continue
            x = c["x"]
            name_s, addr_sort, house, pin = float(x[0]), float(x[12]), float(x[18]), float(x[19])
            empty, short, nblk = float(x[23]) >= 1, float(x[11]) >= 1, float(x[24])
            conflict = float(x[29]) >= 1
            if conflict or (name_s >= 0.75 and addr_sort < 0.25 and not empty):
                patterns["similar_name_address_disagrees"] += 1
            if short and name_s >= 0.8:
                patterns["short_or_common_name"] += 1
            if pin >= 1 and name_s < 0.7:
                patterns["same_pin_weak_name"] += 1
            if house >= 1 and name_s < 0.7:
                patterns["same_house_weak_name"] += 1
            if empty:
                patterns["address_missing"] += 1
            if nblk >= 2:
                patterns["retrieved_by_2plus_blockers"] += 1
            patterns["false_matches"] += 1
    return dict(patterns)


def matrix(rows):
    X, y, groups = [], [], []
    for gi, row in enumerate(rows):
        true = row["true"]
        for c in row["cands"]:
            X.append(c["x"])
            y.append(1 if c["cid"] in true else 0)
            groups.append(gi)
    if not X:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32), np.zeros((0,), np.int8), []
    return np.vstack(X), np.asarray(y, dtype=np.int8), groups


def apply_rule(rows, probas, rule) -> list[set[str]]:
    preds = []
    for row, ps in zip(rows, probas):
        pred = set()
        if not ps:
            preds.append(pred)
            continue
        order = sorted(range(len(ps)), key=lambda i: -ps[i])
        best = ps[order[0]]
        strong_anchor = False
        for rank, i in enumerate(order):
            c = row["cands"][i]
            p = ps[i]
            x = c["x"]
            name_s = float(x[0])
            house = float(x[18]) >= 1.0
            pin = float(x[19]) >= 1.0
            loc = float(x[20]) >= 1.0
            t = rule["t"]
            if float(x[23]) >= 1.0:
                t = max(t, rule["t_empty"])
            if float(x[11]) >= 1.0:
                t = max(t, rule["t_short"])
            if float(x[29]) >= 1.0:
                t = max(t, rule["t_conflict"])
            sibling = (
                rank > 0
                and strong_anchor
                and name_s >= 0.55
                and house
                and (loc or pin)
            )
            if rank > 0 and not sibling:
                t = max(t, rule["t_extra"])
                if best - p > rule["margin"]:
                    continue
            if p >= t or sibling:
                pred.add(c["cid"])
                if name_s >= 0.78 and (house or pin or loc):
                    strong_anchor = True
        preds.append(pred)
    return preds


def proba_of(model, rows, kind: str) -> list[list[float]]:
    out = []
    for row in rows:
        if not row["cands"]:
            out.append([])
            continue
        if kind == "frozen4":
            X = np.vstack([x4_of(c["x"]) for c in row["cands"]])
        else:
            X = np.vstack([c["x"] for c in row["cands"]])
        out.append(model.predict_proba(X)[:, 1].tolist())
    return out


def best_rule(rows, probas) -> tuple[dict, float]:
    best = None
    best_f = -1.0
    for t in [i / 100 for i in range(30, 91, 2)]:
        for t_empty in (t, min(0.95, t + 0.10), min(0.98, t + 0.20)):
            for t_extra in (t, min(0.95, t + 0.08)):
                for t_short in (t, min(0.95, t + 0.10)):
                    for t_conflict in (t, min(0.98, t + 0.15)):
                        for margin in (1.0, 0.12):
                            rule = {
                                "t": t,
                                "t_empty": t_empty,
                                "t_extra": t_extra,
                                "t_short": t_short,
                                "t_conflict": t_conflict,
                                "margin": margin,
                            }
                            f = metrics_of(rows, apply_rule(rows, probas, rule))["macro_f05"]
                            if f > best_f:
                                best_f, best = f, rule
    return best, best_f


def fit_hgb(X, y, sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - pos), 1)
    w = np.where(y == 1, neg / pos, 1.0).astype(np.float32)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype=np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=6, min_samples_leaf=30, random_state=42
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def hard_negative_weights(rows) -> np.ndarray:
    """Upweight same-house weak-name and multi-blocker false pairs."""
    weights = []
    for row in rows:
        true = row["true"]
        for c in row["cands"]:
            if c["cid"] in true:
                weights.append(1.0)
                continue
            x = c["x"]
            name_s = float(x[0])
            house = float(x[18]) >= 1.0
            nblk = float(x[24])
            w = 1.0
            if house and name_s < 0.7:
                w *= 3.0
            if nblk >= 2:
                w *= 2.0
            weights.append(w)
    return np.asarray(weights, dtype=np.float32)


def fit_logreg(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=400, class_weight="balanced", C=1.0)
    clf.fit(Xs, y)
    clf._scaler = scaler
    return clf


def fit_trees(X, y):
    from sklearn.ensemble import ExtraTreesClassifier

    clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2,
    )
    clf.fit(X, y)
    return clf


def proba_logreg(model, rows) -> list[list[float]]:
    out = []
    for row in rows:
        if not row["cands"]:
            out.append([])
            continue
        X = model._scaler.transform(np.vstack([c["x"] for c in row["cands"]]))
        out.append(model.predict_proba(X)[:, 1].tolist())
    return out


def screen(name, model, kind, fit_rows, tune_rows) -> dict:
    if kind == "logreg":
        pt = proba_logreg(model, tune_rows)
    else:
        pt = proba_of(model, tune_rows, kind)
    rule, tune_f = best_rule(tune_rows, pt)
    return {"name": name, "kind": kind, "rule": rule, "tune_macro_f05": tune_f, "model": model}


def main() -> None:
    if CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"loaded {CACHE} n={len(rows)}", flush=True)
    else:
        rows = build_cache()
    split = int(len(rows) * 0.7)
    train, holdout = rows[:split], rows[split:]
    fit_n = int(len(train) * 0.75)
    fit_rows, tune_rows = train[:fit_n], train[fit_n:]
    print(f"fit={len(fit_rows)} tune={len(tune_rows)} holdout={len(holdout)}", flush=True)
    hold_cand = pool_stats(holdout)
    print("holdout_candidates", json.dumps(hold_cand), flush=True)
    if abs(hold_cand["pair_recall"] - 0.8239) > 0.01 or abs(hold_cand["oracle_f05"] - 0.9225) > 0.01:
        raise SystemExit("Candidate set drifted from the verified K=12 lists. Stopping.")

    import joblib

    blob = joblib.load("artifacts/rank_model.joblib")
    frozen, thr = blob["model"], float(blob["threshold"])
    base_pred, base_p = predict_frozen(holdout, frozen, thr)
    baseline = metrics_of(holdout, base_pred)
    print("baseline", json.dumps(baseline), flush=True)
    cats = categorize(holdout, base_pred, base_p)
    contrast = feature_contrast(holdout, base_pred)
    patterns = hard_patterns(holdout, base_pred)
    print("categories", json.dumps(cats["primary"]), flush=True)

    Xf, yf, _ = matrix(fit_rows)
    print(f"fit_pairs={len(yf)} positives={int(yf.sum())}", flush=True)
    screened = []
    print("fit logreg", flush=True)
    screened.append(screen("logreg_rich", fit_logreg(Xf, yf), "logreg", fit_rows, tune_rows))
    print("tune", screened[-1]["name"], screened[-1]["tune_macro_f05"], flush=True)
    print("fit extratrees", flush=True)
    screened.append(screen("extratrees_rich", fit_trees(Xf, yf), "rich", fit_rows, tune_rows))
    print("tune", screened[-1]["name"], screened[-1]["tune_macro_f05"], flush=True)
    print("fit hgb", flush=True)
    screened.append(screen("hgb_rich", fit_hgb(Xf, yf), "rich", fit_rows, tune_rows))
    print("tune", screened[-1]["name"], screened[-1]["tune_macro_f05"], flush=True)

    # Frozen 4-feature model, but the decision rule is chosen on the tune slice only.
    pt = proba_of(frozen, tune_rows, "frozen4")
    rule4, tune4 = best_rule(tune_rows, pt)
    screened.append(
        {
            "name": "frozen4_retuned_rule",
            "kind": "frozen4",
            "rule": rule4,
            "tune_macro_f05": tune4,
            "model": frozen,
        }
    )
    print("tune frozen4_retuned_rule", tune4, flush=True)
    winner = max(screened, key=lambda s: s["tune_macro_f05"])
    print("winner", winner["name"], winner["tune_macro_f05"], winner["rule"], flush=True)

    # Refit the winning family on the whole training portion. Keep the rule chosen
    # on the tune slice. Holdout labels are not used.
    Xtr, ytr, _ = matrix(train)
    if winner["kind"] == "logreg":
        final = fit_logreg(Xtr, ytr)
        hold_p = proba_logreg(final, holdout)
    elif winner["kind"] == "rich":
        final = fit_hgb(Xtr, ytr) if winner["name"].startswith("hgb") else fit_trees(Xtr, ytr)
        hold_p = proba_of(final, holdout, "rich")
    else:
        final = frozen
        hold_p = proba_of(final, holdout, "frozen4")
    hold_pred = apply_rule(holdout, hold_p, winner["rule"])
    hold_metrics = metrics_of(holdout, hold_pred)
    print("holdout_winner", json.dumps(hold_metrics), flush=True)

    # Also score every screened model once on holdout with its tune-chosen rule,
    # using the fit-slice model (no holdout tuning). This is the comparison table.
    comparison = []
    comparison.append({"name": "frozen_threshold_0.64", "features": "4", "rule": {"t": thr}, **baseline})
    for item in screened:
        if item["kind"] == "logreg":
            pp = proba_logreg(item["model"], holdout)
        else:
            pp = proba_of(item["model"], holdout, item["kind"] if item["kind"] != "rich" else "rich")
        m = metrics_of(holdout, apply_rule(holdout, pp, item["rule"]))
        comparison.append(
            {
                "name": item["name"],
                "tune_macro_f05": item["tune_macro_f05"],
                "rule": item["rule"],
                "trained_on": "fit_slice" if item["kind"] != "frozen4" else "original_joblib",
                **m,
            }
        )
    comparison.append(
        {
            "name": winner["name"] + "_refit_train70",
            "tune_macro_f05": winner["tune_macro_f05"],
            "rule": winner["rule"],
            "trained_on": "train70_rule_from_tune",
            **hold_metrics,
        }
    )
    report = {
        "candidate_check": hold_cand,
        "baseline_frozen": baseline,
        "error_categories": cats,
        "feature_contrast_top": contrast["by_feature"][:12],
        "feature_counts": {
            "n_true_in_list": contrast["n_true_in_list"],
            "n_false_selected": contrast["n_false_selected"],
            "n_missed_true_in_list": contrast["n_missed_true_in_list"],
        },
        "hard_negative_patterns": patterns,
        "comparison": [
            {k: v for k, v in row.items() if k != "model"} for row in comparison
        ],
        "winner_name": winner["name"] + "_refit_train70",
        "notes": [
            "Candidate lists are the production K=12 cut.",
            "Thresholds and rules were chosen on the last 25% of the first 70%.",
            "The final 30% was not used to fit or to pick a rule.",
            "LightGBM/XGBoost were not added.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print("WROTE", REPORT, flush=True)


if __name__ == "__main__":
    main()
