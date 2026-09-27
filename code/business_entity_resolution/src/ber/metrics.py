"""F_0.5 scoring matching the challenge definition (macro over S1 entities)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .io_utils import parse_id_list


def f05(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def score_entity(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        # empty pred vs non-empty true → P undefined conventionally → 0 recall, use 0
        # non-empty pred vs empty true → P=0, R=1? Challenge: singleton false merge = 0.0
        return 0.0
    inter = len(pred & true)
    precision = inter / len(pred)
    recall = inter / len(true)
    return f05(precision, recall)


def macro_f05(
    predictions: Mapping[str, set[str]],
    ground_truth: Mapping[str, set[str]],
) -> float:
    scores = []
    for sid, true in ground_truth.items():
        pred = predictions.get(sid, set())
        scores.append(score_entity(pred, true))
    return float(np.mean(scores)) if scores else 0.0


def gt_to_map(df) -> dict[str, set[str]]:
    out = {}
    for row in df.itertuples(index=False):
        out[row.source1_entity_id] = set(parse_id_list(row.matched_entity_ids))
    return out
