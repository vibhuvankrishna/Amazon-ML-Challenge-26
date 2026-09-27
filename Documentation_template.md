# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary

We resolve business records across three noisy sources with a two-stage pipeline: a small inverted-index blocking step, then a precision-oriented string scorer. Source 1 is the deduplicated reference. For each Source 1 entity we keep at most 8 Source 2 / Source 3 candidates, then accept a match only when a weighted name-and-address score clears a high threshold. The design targets macro F₀.₅ (false merges hurt more than misses) and a small candidate set per Source 1 entity, because final ranking reviews blocking size as well as the leaderboard score.

---

## 2. Methodology

### 2.1 Problem Analysis

Training covers the US and India. The test set also contains France, so country is treated as an open string label and is never one-hot encoded or filtered to the training countries. Names vary by legal suffix (Corp / Corporation, Pvt / Private, Ltd / Limited), punctuation, word order, typos, and transliteration. Addresses vary by abbreviation (Rd / Road), missing components, reordered tokens, and landmark text. Some Source 2 and Source 3 addresses are empty. A Source 1 entity may match zero, one, or many Source 2 / Source 3 records. On a training sample, most entities have a few matches and a minority are singletons. Correct empty predictions score 1.0 under macro F₀.₅; any false match on a singleton scores 0.0.

Scale rules out all-pairs comparison. Training is about 2.2 million Source 1 records against about 10 million Source 2 + Source 3 records. Test is about 1.73 million Source 1 records against about 10 million Source 2 + Source 3 records.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise scorer (hybrid rules and string similarity)

**Core Innovation:** Selective blocking keys (two-token name key and PIN/ZIP) with a hard cap of 8 candidates per Source 1 entity, then a single weighted similarity score tuned for precision. Multi-key hits are preferred inside that cap so the candidate file stays small without dropping the strongest overlaps.

Pipeline:

1. Normalize names and addresses with local rules only (no geocoding, no external business lookup).
2. Build an inverted index over Source 2 and Source 3.
3. Retrieve at most 8 candidates per Source 1 entity. This list is written to `candidate_pairs.tsv`.
4. Score each candidate. Keep IDs whose score is at least 0.84. This list is written to `matching_results.tsv` and is always a subset of the candidates.

---

## 3. Candidate Generation (Blocking)

Blocking keys (country is always part of the key, so cross-country pairs are not generated):

- `country + first two significant name tokens` (primary key)
- `country + first significant name token` when the name has only one token
- `country + first 5+ digit run in the address` (PIN / ZIP), at most one such key

Common legal words are stripped before keying (corporation, limited, private, and similar suffixes) so they do not explode the index. Each posting list is capped (8 IDs) so popular keys cannot create huge candidate sets.

- **Blocking keys used:** country-scoped name bigram, name unigram fallback, PIN/ZIP digit
- **Candidate cap:** at most 8 Source 2 / Source 3 IDs per Source 1 entity
- **How true matches are retained:** a true match that shares the normalized name bigram or a long digit token lands in the posting list; within the cap, records that hit more than one key are kept first

`candidate_pairs.tsv` is exactly this final candidate list — the set scored by the matcher — not an earlier, larger blocking pass.

---

## 4. Matching Model

**Features used:**

- Name features: normalized-string ratio (RapidFuzz), Jaro–Winkler, token Jaccard after suffix stripping
- Address features: Jaccard overlap of digit tokens (house number / PIN / ZIP)
- Other: hard country equality gate (score is 0 if countries differ); a boost when digit overlap is high and the name ratio is already strong

**Model type:** Weighted linear score over those similarities (weights 0.45 ratio, 0.30 Jaro–Winkler, 0.20 token Jaccard, 0.05 digit Jaccard). No external API and no model above the license or parameter limits. An earlier HistGradientBoosting classifier on the same feature family was trained on a 25,000-entity training sample (validation macro F₀.₅ about 0.89) and is kept in `artifacts/` for comparison. Full-test inference uses the weighted scorer so the run stays inside laptop memory and keeps the candidate cap small.

**Threshold selection method:** Threshold 0.84, chosen on the precision-heavy side of the F₀.₅ curve. Empty predictions are preferred over weak merges, because a false merge on a singleton scores 0 and F₀.₅ weights precision twice as much as recall.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** about 0.89 on a 25,000 Source 1 training holdout with the gradient-boosted pair classifier and a larger candidate cap. The submitted full-test run uses the weighted scorer and an 8-candidate cap; that exact full-test number is the portal score.
- **Common false positives (wrong merges):** businesses that share a short distinctive token and a city or PIN but are different legal entities; very common first tokens if the bigram key is weak.
- **Common false negatives (missed matches):** heavy typos that change both leading name tokens, trade names that do not overlap the legal name, and rows whose only shared signal is a reordered address with no long digit token.

---

## 6. Conclusion

Entity resolution at this scale is a blocking problem first and a matching problem second. Capping candidates at 8 per Source 1 entity keeps the comparison space small enough to rank well on blocking quality, and a high score threshold protects macro F₀.₅ from false merges. The same code path writes both submission files, so every accepted match is a candidate the model actually scored.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
├── scripts/smoke_test.sh
└── src/
    ├── train.py          # optional CPU classifier on a training sample
    ├── infer.py          # memory-safe SQLite indexer
    ├── infer_fast.py     # entry point that writes both output TSVs
    └── ber/
        ├── normalize.py
        ├── blocking.py
        ├── features.py
        ├── metrics.py
        └── io_utils.py
```

Reproduce the two output files from the `student_resource/` directory:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
export PYTHONPATH=code/business_entity_resolution/src
python code/business_entity_resolution/src/infer_fast.py \
  --test-dir dataset/test \
  --db artifacts/test_s23.sqlite \
  --output-dir output \
  --threshold 0.84 \
  --max-candidates 8
```

`artifacts/test_s23.sqlite` is the local index of test Source 2 and Source 3 built by the pipeline. If it is absent, `src/infer.py` rebuilds it from `dataset/test/test_source2.tsv` and `test_source3.tsv`.

Validate before upload:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### B. Additional Results

- Training sample used for the gradient-boosted reference model: 25,000 Source 1 entities, validation macro F₀.₅ 0.893, threshold 0.92, blocking pair recall about 0.80 at a cap of 120 candidates.
- Submitted run tightens that cap to 8 candidates so the reviewed candidate file stays small.
- No external databases, geocoders, or commercial entity-resolution APIs are used.

---

**Note:** Replace the team-name placeholders before zipping. Sections describe the submitted pipeline.
