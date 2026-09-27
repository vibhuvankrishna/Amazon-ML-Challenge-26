"""Train a cheap sklearn pair classifier — CPU-only, credit-safe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ber.blocking import (
    blocking_recall,
    build_index,
    expand_pairs,
    generate_candidates_from_index,
    records_by_id,
)
from ber.features import FEATURE_NAMES, build_feature_matrix
from ber.io_utils import iter_chunks, parse_id_list, read_ground_truth, read_source
from ber.metrics import gt_to_map, macro_f05


def load_training_slice(
    data_dir: Path,
    sample: int,
    extra_s23: int = 120_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    if sample <= 0:
        print("Loading FULL train (this needs RAM — prefer --sample first)…")
        s1 = read_source(data_dir / "train_source1.tsv")
        s2 = read_source(data_dir / "train_source2.tsv")
        s3 = read_source(data_dir / "train_source3.tsv")
        gt = read_ground_truth(data_dir / "train_ground_truth.tsv")
        return s1, s2, s3, gt

    print(f"Sample mode: {sample} S1 entities (coherent GT + match fetch)…")
    s1 = read_source(data_dir / "train_source1.tsv", nrows=sample)
    s1_ids = set(s1["entity_id"])

    gt_all = read_ground_truth(data_dir / "train_ground_truth.tsv")
    gt = gt_all[gt_all["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)
    print(f"  GT rows for sample: {len(gt):,}")

    need: set[str] = set()
    for raw in gt["matched_entity_ids"]:
        need.update(parse_id_list(raw))
    print(f"  Required S2/S3 match IDs: {len(need):,}")

    def fetch_source(path: Path, prefix: str) -> pd.DataFrame:
        want = {i for i in need if i.startswith(prefix)}
        kept, extras = [], []
        n_extra = 0
        rng = np.random.default_rng(42)
        for chunk in iter_chunks(path, chunksize=250_000):
            hit = chunk[chunk["entity_id"].isin(want)]
            if len(hit):
                kept.append(hit)
                want -= set(hit["entity_id"])
            if n_extra < extra_s23:
                n_take = min(4_000, len(chunk), extra_s23 - n_extra)
                if n_take > 0:
                    take = chunk.sample(n=n_take, random_state=int(rng.integers(1e9)))
                    extras.append(take)
                    n_extra += len(take)
            if not want and n_extra >= extra_s23:
                break
        parts = kept + extras
        if not parts:
            return read_source(path, nrows=extra_s23)
        out = pd.concat(parts, ignore_index=True).drop_duplicates("entity_id")
        missing = {i for i in need if i.startswith(prefix)} - set(out["entity_id"])
        if missing:
            print(f"  WARN: {len(missing)} {prefix} match IDs not found in scan")
        return out

    s2 = fetch_source(data_dir / "train_source2.tsv", "S2-")
    s3 = fetch_source(data_dir / "train_source3.tsv", "S3-")
    print(f"  Loaded S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3, gt


def build_labeled_pairs(
    s1: pd.DataFrame,
    index: dict,
    s23: pd.DataFrame,
    s23_rec: dict,
    gt_map: dict[str, set[str]],
    max_candidates: int,
    max_neg_per_pos: int = 5,
) -> tuple[pd.DataFrame, dict]:
    candidates = generate_candidates_from_index(s1, index, max_candidates=max_candidates)
    br = blocking_recall(candidates, {sid: gt_map.get(sid, set()) for sid in s1["entity_id"]})
    s1_rec = records_by_id(s1)
    pairs = expand_pairs(candidates, s1, s23, s1_rec=s1_rec, s23_rec=s23_rec)
    if pairs.empty:
        return pd.DataFrame(columns=["label"]), br

    labels, keep_idx = [], []
    rng = np.random.default_rng(42)
    for sid, group in pairs.groupby("source1_entity_id"):
        true = gt_map.get(sid, set())
        pos_idx, neg_idx = [], []
        for i, cid in zip(group.index, group["candidate_entity_id"]):
            (pos_idx if cid in true else neg_idx).append(i)
        keep = list(pos_idx)
        n_neg = min(len(neg_idx), max(len(pos_idx) * max_neg_per_pos, max_neg_per_pos))
        if neg_idx and n_neg > 0:
            keep.extend(rng.choice(neg_idx, size=n_neg, replace=False).tolist())
        for i in keep:
            labels.append(1 if pairs.at[i, "candidate_entity_id"] in true else 0)
            keep_idx.append(i)
    if not keep_idx:
        out = pairs.iloc[0:0].copy()
        out["label"] = []
        return out, br
    out = pairs.loc[keep_idx].copy()
    out["label"] = labels
    return out.reset_index(drop=True), br


def predict_map_from_proba(
    pairs: pd.DataFrame, proba: np.ndarray, threshold: float, s1_ids: list[str]
):
    pred = {sid: set() for sid in s1_ids}
    if pairs.empty:
        return pred
    tmp = pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    tmp["proba"] = proba
    kept = tmp[tmp["proba"] >= threshold]
    for sid, group in kept.groupby("source1_entity_id"):
        pred[sid] = set(group["candidate_entity_id"].tolist())
    return pred


def tune_threshold(model, pairs, gt_map, s1_ids) -> tuple[float, float]:
    """Score features once, sweep thresholds (avoids 30× rapidfuzz rebuild)."""
    if pairs.empty:
        return 0.5, 0.0
    print(f"  building features for {len(pairs):,} val pairs (once)…")
    X = build_feature_matrix(pairs)
    proba = model.predict_proba(X)[:, 1]
    sub_gt = {k: gt_map[k] for k in s1_ids if k in gt_map}
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.30, 0.92, 32):
        preds = predict_map_from_proba(pairs, proba, float(t), s1_ids)
        s = macro_f05(preds, sub_gt)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def main() -> None:
    p = argparse.ArgumentParser(description="Train pair matcher (local CPU)")
    p.add_argument("--data-dir", default="dataset/train")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--max-per-key", type=int, default=250)
    p.add_argument("--extra-s23", type=int, default=120_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    s1, s2, s3, gt = load_training_slice(
        Path(args.data_dir), args.sample, extra_s23=args.extra_s23
    )
    gt_map = gt_to_map(gt)
    s1 = s1[s1["entity_id"].isin(gt_map)].reset_index(drop=True)
    print(f"S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    if len(s1) < 10:
        raise SystemExit("Too few S1 rows after GT align.")

    s23 = pd.concat([s2, s3], ignore_index=True)
    print("Building shared blocking index…")
    index = build_index(s23, max_per_key=args.max_per_key)
    s23_rec = records_by_id(s23)
    print(f"Index keys: {len(index):,}")

    s1_train, s1_val = train_test_split(s1, test_size=args.val_frac, random_state=args.seed)
    s1_train = s1_train.reset_index(drop=True)
    s1_val = s1_val.reset_index(drop=True)
    print(f"Split: train S1={len(s1_train):,}  val S1={len(s1_val):,}")

    print("Building labeled candidate pairs (train)…")
    train_pairs, br_train = build_labeled_pairs(
        s1_train, index, s23, s23_rec, gt_map, args.max_candidates
    )
    print(
        f"  blocking pair_recall={br_train['pair_recall']:.3f}  "
        f"entity_full={br_train['entity_full_recall']:.3f}  "
        f"avg_cands={br_train['avg_candidates']:.1f}"
    )
    if train_pairs.empty or train_pairs["label"].sum() == 0:
        raise SystemExit("No positive training pairs — raise --max-candidates / --extra-s23.")
    print(f"Train pairs: {len(train_pairs):,}  positives={int(train_pairs['label'].sum()):,}")

    X = build_feature_matrix(train_pairs)
    y = train_pairs["label"].to_numpy()
    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=20,
        random_state=args.seed,
    )
    print("Fitting HistGradientBoosting on CPU…")
    model.fit(X, y)

    print("Val candidates + threshold tune…")
    val_cand = generate_candidates_from_index(s1_val, index, max_candidates=args.max_candidates)
    br_val = blocking_recall(val_cand, {sid: gt_map.get(sid, set()) for sid in s1_val["entity_id"]})
    print(
        f"  val blocking pair_recall={br_val['pair_recall']:.3f}  "
        f"entity_full={br_val['entity_full_recall']:.3f}"
    )
    val_pairs = expand_pairs(
        val_cand, s1_val, s23, s1_rec=records_by_id(s1_val), s23_rec=s23_rec
    )
    thr, val_score = tune_threshold(model, val_pairs, gt_map, s1_val["entity_id"].tolist())
    print(f"Best threshold={thr:.3f}  val macro-F0.5={val_score:.4f}")

    joblib.dump(
        {
            "model": model,
            "threshold": thr,
            "features": FEATURE_NAMES,
            "max_candidates": args.max_candidates,
            "max_per_key": args.max_per_key,
        },
        art / "model.joblib",
    )
    meta = {
        "threshold": thr,
        "val_macro_f05": val_score,
        "sample": args.sample,
        "max_candidates": args.max_candidates,
        "blocking_train": br_train,
        "blocking_val": br_val,
        "n_train_pairs": int(len(train_pairs)),
        "n_val_s1": int(len(s1_val)),
        "model": "HistGradientBoostingClassifier",
    }
    (art / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {art / 'model.joblib'}")


if __name__ == "__main__":
    main()
