# Business Entity Resolution — Amazon ML Challenge 2026

**Credit-safe default: run everything on your laptop.** The dataset is already local. Use AWS/SageMaker only if your machine cannot finish a full pass.

## Why local-first?

| Approach | Cost | Recommendation |
| --- | --- | --- |
| Laptop CPU (this repo) | **$0 credits** | Default |
| SageMaker `ml.t3.medium` notebook | ~free-tier / small credit burn | Only if laptop is too slow/RAM-limited |
| SageMaker Training Jobs / GPU / Endpoints | Burns credits fast | **Avoid** for this challenge |

You submit **TSV files**, not a hosted API — never deploy an endpoint.

## Setup (local)

```bash
cd student_resource
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Raise-score path (HGB v3 + sibling accept)

Production decision model: `artifacts/rank_model_hgb_v3.joblib` (rich 30-feature HGB, rule cutoffs from holdout ~0.83).

```bash
# Unit checks (no full dataset)
python artifacts/analysis/test_raise_score.py

# Refit with hard-negative weights when K=12 cache / train indexes exist
python artifacts/analysis/raise_score.py --min-f05 0.83

# Full test infer → output_v6/  (needs dataset/test + artifacts/test_s23.sqlite)
python artifacts/analysis/run_output_v6.py

python utils/validate_submission.py \
  --matching output_v6/matching_results.tsv \
  --candidate output_v6/candidate_pairs.tsv \
  --test-dir dataset/test
```

Changes vs the 0.42 string scorer: sibling acceptance for multi-source matches, typo/short-name blocking keys (`rare_pin`, `phonetic_house`, `sorted_house`), French address expansions, hard-negative upweighting in refits.

## Commands

### Smoke test (minutes, tiny sample)

```bash
bash code/business_entity_resolution/scripts/smoke_test.sh
```

### Train (scale up sample → full)

```bash
export PYTHONPATH=code/business_entity_resolution/src

# Scaled local train (recommended; $0 credits)
python code/business_entity_resolution/src/train.py --sample 25000 --max-candidates 120 --extra-s23 180000

# Larger overnight train
python code/business_entity_resolution/src/train.py --sample 100000 --max-candidates 120
```

### Infer on test

```bash
# Full test — local overnight job (no AWS)
bash code/business_entity_resolution/scripts/run_full_infer.sh
# or:
PYTHONUNBUFFERED=1 python -u code/business_entity_resolution/src/infer.py --sample 0 --max-candidates 80 --chunk-s1 8000
```

Check progress: `tail -f logs/full_infer_*.log`

### Validate before portal upload

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## If you must use AWS (limited credits)

1. Create **one** Notebook Instance: `ml.t3.medium` only  
2. Upload code + (optionally) data to S3, or clone and sync  
3. Run the same train/infer commands inside the notebook terminal  
4. Download `output/*.tsv`  
5. **Stop** the notebook immediately when idle (do not delete mid-challenge)  
6. Never create endpoints, GPU instances, or multiple notebooks  

Set a billing alarm on Day 0.

## Layout

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
├── scripts/smoke_test.sh
└── src/
    ├── train.py
    ├── infer.py
    └── ber/
        ├── io_utils.py
        ├── normalize.py
        ├── blocking.py
        ├── features.py
        └── metrics.py
```

Artifacts land in repo-root `artifacts/`; submissions in `output/`.
