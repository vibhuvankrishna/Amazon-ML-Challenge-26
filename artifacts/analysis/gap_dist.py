"""Read-only train/test/output_v4 distribution diagnostic. Does not write submissions."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

csv.field_size_limit(10_000_000)
ROOT = Path("/Users/m.harshithakrishna/Desktop/student_resource")
OUT = ROOT / "artifacts/analysis/gap_dist.json"
DIGIT = re.compile(r"\d+")
DEV = str.maketrans("०१२३४५६७८९", "0123456789")
WORD = re.compile(r"[^\W\d_]{6,}", re.UNICODE)


def country(value: str) -> str:
    return (value or "").strip().upper() or "UNK"


def text_flags(name: str, addr: str) -> dict:
    name = name or ""
    addr = addr or ""
    runs = DIGIT.findall(addr.translate(DEV))
    toks = [t for t in name.split() if t]
    non_ascii = any(ord(c) > 127 for c in name)
    dev = any("\u0900" <= c <= "\u097F" for c in name)
    accent = False
    if non_ascii and not dev:
        folded = unicodedata.normalize("NFKD", name)
        accent = any(unicodedata.combining(c) for c in folded) or any(ord(c) > 127 for c in name)
    return {
        "name_len": len(name),
        "addr_len": len(addr),
        "toks": len(toks),
        "addr_empty": int(not addr.strip()),
        "house": int(any(1 <= len(d) <= 6 for d in runs)),
        "pin": int(any(len(d) >= 5 for d in runs)),
        "loc": int(bool(WORD.search(addr))),
        "non_ascii": int(non_ascii),
        "devanagari": int(dev),
        "latin_ext": int(accent and not dev),
        "punct": int(any(not c.isalnum() and not c.isspace() for c in name)),
        "short": int(len(toks) <= 1 or len(name) < 8),
    }


def name_bin(n: int) -> str:
    if n < 8:
        return "0-7"
    if n < 16:
        return "8-15"
    if n < 31:
        return "16-30"
    if n < 61:
        return "31-60"
    return "61+"


class Acc:
    def __init__(self):
        self.n = 0
        self.s = Counter()

    def add(self, flags: dict):
        self.n += 1
        for k, v in flags.items():
            self.s[k] += v

    def rates(self) -> dict:
        n = max(self.n, 1)
        out = {"n": self.n}
        for k, v in self.s.items():
            if k in {"name_len", "addr_len", "toks"}:
                out[k] = round(v / n, 3)
            else:
                out[k] = round(v / n, 4)
        return out


def stream_train():
    acc = Acc()
    countries = Counter()
    bins = Counter()
    path = ROOT / "dataset/train/train_source1.tsv"
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            flags = text_flags(row.get("business_name") or "", row.get("business_address") or "")
            acc.add(flags)
            countries[country(row.get("country") or "")] += 1
            bins[name_bin(flags["name_len"])] += 1
            if i and i % 500_000 == 0:
                print(f"train {i}", flush=True)
    return acc, countries, bins


def stream_test():
    s1 = ROOT / "dataset/test/test_source1.tsv"
    match = ROOT / "output_v4/matching_results.tsv"
    cand = ROOT / "output_v4/candidate_pairs.tsv"
    acc = Acc()
    bins = Counter()
    match_hist = Counter()
    cand_hist = Counter()
    by_c = {}
    bin_pred = {}
    addr_pred = {"empty_addr": Counter(), "has_addr": Counter()}
    mismatches = 0
    n = 0
    with s1.open(newline="") as a, match.open(newline="") as b, cand.open(newline="") as c:
        ra = csv.DictReader(a, delimiter="\t")
        rb = csv.DictReader(b, delimiter="\t")
        rc = csv.DictReader(c, delimiter="\t")
        for row, mr, cr in zip(ra, rb, rc):
            sid = row["entity_id"]
            if sid != mr["source1_entity_id"] or sid != cr["source1_entity_id"]:
                mismatches += 1
                if mismatches > 20:
                    break
                continue
            flags = text_flags(row.get("business_name") or "", row.get("business_address") or "")
            acc.add(flags)
            bins[name_bin(flags["name_len"])] += 1
            mids = [x for x in (mr.get("matched_entity_ids") or "").split(",") if x]
            cids = [x for x in (cr.get("candidate_entity_ids") or "").split(",") if x]
            nm, nc = len(mids), len(cids)
            match_hist[nm] += 1
            cand_hist[min(nc, 12)] += 1
            cc = country(row.get("country") or "")
            slot = by_c.get(cc)
            if slot is None:
                slot = {
                    "n": 0,
                    "empty_pred": 0,
                    "one": 0,
                    "multi": 0,
                    "zero_cand": 0,
                    "reject_all": 0,
                    "saturated": 0,
                    "sum_pred": 0,
                    "sum_cand": 0,
                    "flags": Counter(),
                }
                by_c[cc] = slot
            slot["n"] += 1
            slot["sum_pred"] += nm
            slot["sum_cand"] += nc
            slot["empty_pred"] += int(nm == 0)
            slot["one"] += int(nm == 1)
            slot["multi"] += int(nm >= 2)
            slot["zero_cand"] += int(nc == 0)
            slot["reject_all"] += int(nc > 0 and nm == 0)
            slot["saturated"] += int(nc >= 12)
            for k, v in flags.items():
                slot["flags"][k] += v
            bkey = name_bin(flags["name_len"])
            bp = bin_pred.setdefault(bkey, Counter())
            bp["n"] += 1
            bp["empty"] += int(nm == 0)
            bp["zero_cand"] += int(nc == 0)
            bp["reject"] += int(nc > 0 and nm == 0)
            ak = "empty_addr" if flags["addr_empty"] else "has_addr"
            addr_pred[ak]["n"] += 1
            addr_pred[ak]["empty"] += int(nm == 0)
            addr_pred[ak]["zero_cand"] += int(nc == 0)
            addr_pred[ak]["reject"] += int(nc > 0 and nm == 0)
            n += 1
            if n % 400_000 == 0:
                print(f"test {n}", flush=True)
        extra_m = sum(1 for _ in rb)
        extra_c = sum(1 for _ in rc)
        extra_s = sum(1 for _ in ra)
    countries = []
    for cc, slot in by_c.items():
        nn = slot["n"]
        countries.append(
            {
                "country": cc,
                "n": nn,
                "share": round(nn / max(n, 1), 4),
                "empty_pred": round(slot["empty_pred"] / nn, 4),
                "one": round(slot["one"] / nn, 4),
                "multi": round(slot["multi"] / nn, 4),
                "zero_cand": round(slot["zero_cand"] / nn, 4),
                "reject_all": round(slot["reject_all"] / nn, 4),
                "saturated_k12": round(slot["saturated"] / nn, 4),
                "avg_pred": round(slot["sum_pred"] / nn, 3),
                "avg_cand": round(slot["sum_cand"] / nn, 3),
                "addr_empty": round(slot["flags"]["addr_empty"] / nn, 4),
                "non_ascii": round(slot["flags"]["non_ascii"] / nn, 4),
                "devanagari": round(slot["flags"]["devanagari"] / nn, 4),
                "latin_ext": round(slot["flags"]["latin_ext"] / nn, 4),
                "house": round(slot["flags"]["house"] / nn, 4),
                "pin": round(slot["flags"]["pin"] / nn, 4),
                "short": round(slot["flags"]["short"] / nn, 4),
                "name_len": round(slot["flags"]["name_len"] / nn, 2),
            }
        )
    countries.sort(key=lambda r: -r["n"])
    return {
        "rows": n,
        "id_mismatches": mismatches,
        "extra_match_rows": extra_m,
        "extra_cand_rows": extra_c,
        "extra_s1_rows": extra_s,
        "text": acc.rates(),
        "name_bins": dict(bins),
        "match_hist": {str(k): v for k, v in sorted(match_hist.items())},
        "cand_hist": {str(k): v for k, v in sorted(cand_hist.items())},
        "by_name_bin": {k: dict(v) for k, v in bin_pred.items()},
        "by_address": {k: dict(v) for k, v in addr_pred.items()},
        "countries": countries,
    }


def main():
    print("train pass", flush=True)
    tr_acc, tr_c, tr_bins = stream_train()
    print("test pass", flush=True)
    test = stream_test()
    train_names = set(tr_c)
    test_names = {r["country"] for r in test["countries"]}
    only_test = []
    for row in test["countries"]:
        if row["country"] not in train_names:
            only_test.append({"country": row["country"], "n": row["n"], "share": row["share"]})
    only_test.sort(key=lambda r: -r["n"])
    train_top = [
        {"country": k, "n": v, "share": round(v / tr_acc.n, 4)}
        for k, v in tr_c.most_common(25)
    ]
    payload = {
        "train": {
            "n": tr_acc.n,
            "text": tr_acc.rates(),
            "name_bins": dict(tr_bins),
            "n_countries": len(tr_c),
            "top_countries": train_top,
        },
        "test": test,
        "countries_only_in_test": only_test,
        "only_in_test_rows": sum(r["n"] for r in only_test),
        "only_in_test_share": round(sum(r["n"] for r in only_test) / max(test["rows"], 1), 4),
    }
    OUT.write_text(json.dumps(payload))
    print("WROTE", OUT, "train", tr_acc.n, "test", test["rows"], "only_test_countries", len(only_test), flush=True)
    print("ONLY_TEST", only_test[:15], flush=True)
    print("TEST_TOP", test["countries"][:12], flush=True)


if __name__ == "__main__":
    main()
