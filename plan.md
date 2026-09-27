# Amazon ML Challenge 2026 — Full Plan

**Challenge window:** 25 Sep 2026, 12:00 AM IST → 27 Sep 2026, 11:59 PM IST (72 hours)  
**Task type:** Business Entity Resolution (match noisy business records across 3 sources)  
**Metric:** Macro-averaged **F₀.₅** (precision-heavy)  
**Submission cap:** **5 leaderboard uploads per day**

---

## 1. What the problem is (plain English)

Amazon gets business identity data from many independent systems. The same real shop/company shows up in multiple files with **different IDs**, messy names, and messy addresses — and **no shared key** to join them.

Your job is **entity resolution**:

> For every record in **Source 1** (the clean reference list), find **all** matching records in **Source 2** and **Source 3**. A Source-1 business can match **0, 1, or many** records.

This is **not** classification of a single row. It is **pairwise matching at massive scale**, so you cannot compare every record to every other record. You must:

1. **Block / generate candidates** (cheap filters → small set of plausible matches)
2. **Score / classify** those candidates (ML or similarity model → keep/reject)
3. Output final matches + the candidate set you scored

**Why F₀.₅?** Wrong merges (linking two different businesses) hurt more than missed matches. So prefer **being careful** over matching everything.

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed **per Source-1 entity**, then **averaged** (macro). Correct empty predictions on singletons score **1.0**; any false match on a singleton scores **0.0**.

---

## 2. What’s in this folder (attached / provided files)

| File / folder | What it is | How you use it |
| --- | --- | --- |
| `README.md` | Full problem statement (same content as the PDF, easier to search) | Primary reference while coding |
| `problem statement.pdf` | Official problem PDF + note that **blocking quality counts in final ranking** | Read once; keep open for output rules |
| `guidelines.pdf` | Hackathon rules, timeline, submission limits, AWS Builder prep link | Read for logistics + credits path |
| `Documentation_template.md` | Methodology write-up template | Fill this for the final zip |
| `utils/validate_submission.py` | Format checker (stdlib only) | Run **before every** portal upload |
| `dataset/train/*.tsv` | Labeled training data | EDA, train model, local validation |
| `dataset/test/*.tsv` | Unlabeled test data | Produce final predictions |
| Email screenshots | Unstop / Team Amazon emails | Confirm dataset is **team-leader only** to download; share within team |

### Dataset columns (all sources)

Every `*_source1/2/3.tsv` has:

- `entity_id` — unique; prefix `S1-` / `S2-` / `S3-` tells the source
- `business_name` — noisy (abbr, typos, Hindi/English, legal suffixes)
- `business_address` — noisy (abbr, missing PIN/state, landmarks, reorder)
- `country` — string label; **train = US + India**; **test also has France** (never seen in train)

Ground truth (`train_ground_truth.tsv`):

- `source1_entity_id`
- `matched_entity_ids` — comma-separated S2/S3 IDs (empty = singleton)

**Always read with tab separator:**

```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

### Approximate scale (from your local files)

| Split | Source 1 | Source 2 | Source 3 | Ground truth |
| --- | ---: | ---: | ---: | ---: |
| Train | ~2.21M | ~5.03M | ~5.29M | ~2.21M rows |
| Test | ~1.73M | ~4.89M | ~5.08M | none |

Naive all-pairs on train is on the order of **tens of trillions** of comparisons → **blocking is mandatory**.

Sample EDA notes (first ~8k–50k rows):

- Train countries roughly ~60% US / ~40% India
- Test Source 1 includes **France** (~15% in sample) — do **not** hard-code only US/India
- Some S2/S3 addresses are empty (~3–4% in sample)
- In a 50k GT sample: ~5.6% singletons, ~3.5 matches per S1 on average (max seen ~11)

### Expected outputs

**During challenge (leaderboard):** upload only `matching_results.tsv`

| Column | Meaning |
| --- | --- |
| `source1_entity_id` | Every S1 test ID, exactly once |
| `matched_entity_ids` | Comma-separated S2/S3 IDs, or empty |

**Final zip (all teams):**

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv      # last blocking stage fed to the model
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md    # filled in
```

**Final ranking is not only F₀.₅.** Judges also review `candidate_pairs.tsv`. A **smaller candidate set per Source 1 entity** ranks higher, as long as true matches still sit inside that set. Target about **5–12 candidates per S1**, not hundreds. `matching_results.tsv` must be a subset of those candidates.

Validate locally:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Hard rules (disqualification risk)

- **No external lookup:** no Google/Maps geocoding APIs, no company registries, no commercial ER APIs, no scraping the web to “find the real business”
- Models: **MIT / Apache 2.0**, **≤ 8B parameters**
- Desktop/laptop only; no simultaneous multi-device logins

---

## 3. AWS Builder Center — what it is and how it fits

Official prep guide (linked from `guidelines.pdf`):

https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo

**AWS Builder Center** (`builder.aws.com`) is Amazon’s student/builder community + learning hub. For this challenge you use it for:

| Use | Why it matters |
| --- | --- |
| **Builder Profile ID** | Required to register for the challenge |
| **Prep guide + Twitch recording** | Live SageMaker demo, best practices, credit tips |
| **Free sandbox** | Real AWS practice, **no credit card**, ~8 hrs/week |
| **Student Rewards badges** | Extra free credits / Skill Builder / cert voucher value |
| **Workshops** | Hands-on AWS + ML learning (not the competition dataset) |

**Setup (≈5 minutes, no credit card for Builder profile):**

1. Go to [builder.aws.com](https://builder.aws.com) → Join → verify email → pick alias  
2. Keep your **Builder Profile ID** handy (registration requirement)  
3. Watch the prep recording on Twitch (linked in the guide)  
4. Optionally earn Student Rewards badges for extra credits

Builder Center is **not** where you submit challenge answers. Submissions go through the **Unstop / challenge portal**. Builder = prep + AWS identity + learning; **AWS account + SageMaker/S3** = compute for training.

---

## 4. AWS credits — what you get and how to use them

From the official AWS Builder prep guide (Sep 2026):

| Credit | Who | Notes |
| --- | --- | --- |
| **$200 AWS credits** | Every participant | Via Free Tier / challenge signup path |
| **Extra $100** | Top 500 teams at ~48-hour mark | Speed + quality incentive |
| Student Rewards | Optional | Badges → small extra credits / Skill Builder / exam voucher |

**How the $200 typically unlocks (per guide):**

1. Create / use an **AWS account** (Free Tier path)
2. You get **$100** on signup
3. Complete **5 getting-started activities** (e.g. create an EC2 instance, RDS, etc.) → another **$100** → **$200 total**

**What to spend credits on for THIS challenge:**

| Service | Role in your pipeline |
| --- | --- |
| **Amazon S3** | Store train/test TSVs, embeddings, models, outputs |
| **SageMaker Notebook Instance** (`ml.t3.medium`) | Interactive coding / EDA / local training in notebook |
| **SageMaker Training Jobs** (optional) | Bigger CPU/GPU machines if embeddings / deep models need them |
| **EC2** (optional) | Raw VM if you prefer custom env over SageMaker |
| **Avoid long-lived Endpoints** | Endpoints bill ~$0.12/hr even idle — **not needed**; you submit TSV files, not an API |

**Recommended compute pattern for the hackathon (from AWS guide):**

> **Notebook Instance + train locally in the notebook + predict locally → write TSV.**  
> Do **not** deploy an endpoint for scoring.

**Concrete AWS workflow for this ER problem:**

1. Create notebook: SageMaker → Notebook instances → `ml.t3.medium` → new IAM role → wait until `InService` → Open JupyterLab  
2. Upload dataset to **your S3 bucket** (or sync from local)  
3. Run blocking + matching pipeline in the notebook / scripts  
4. Write `matching_results.tsv` + `candidate_pairs.tsv`  
5. Download outputs → validate → upload to Unstop portal  
6. **Stop** the notebook when idle (stops billing; files persist). Delete only after the challenge.

**Cost hygiene (do this Day 0):**

- Set a **billing alert** in AWS Billing immediately  
- Prefer `us-east-1`  
- Stop notebooks when not working  
- Never leave endpoints running  
- Prefer free-tier-covered instance types first (`ml.t3.medium`)

**Fair-play reminder:** AWS is for **compute and storage of the provided data**. Using AWS APIs that look up external business identity / geocoding still violates the “no external data lookup” rule.

---

## 5. Recommended technical approach (72-hour plan)

### High-level architecture

```
Normalize names/addresses
        ↓
Blocking (country + phonetic / n-grams / locality tokens / ANN)
        ↓
candidate_pairs.tsv  ← save this exact set
        ↓
Pair features (string sims, token overlap, country match, …)
        ↓
Classifier / ranker (XGBoost / LightGBM / logistic / small transformer)
        ↓
Threshold tuned for F_0.5 on validation
        ↓
matching_results.tsv
```

### Day-by-day plan

#### Day 0 / Hour 0–2 — Setup

- [ ] Confirm team leader shared full `dataset/` with everyone  
- [ ] Create AWS account + redeem / unlock credits; create SageMaker notebook; set billing alert  
- [ ] Create Builder Center profile if missing; skim prep guide + Twitch recording  
- [ ] Repo layout: `src/`, `output/`, `notebooks/`, pin `requirements.txt`  
- [ ] Smoke-test: load small sample with `sep="\t"`; run validator on a dummy output  

#### Day 1 — EDA + strong blocking + baseline

- [ ] Country distribution (especially France in test)  
- [ ] Match-count distribution / singleton rate from GT  
- [ ] Name/address noise patterns (Hindi vs English, Corp/Ltd, Rd/St, empty address)  
- [ ] **Normalization:** lowercasing, strip punctuation, expand Corp↔Corporation, Pvt↔Private, Rd↔Road, remove legal suffixes for fuzzy keys, keep original for features  
- [ ] **Blocking ideas (combine several):**  
  - Exact `country` (always)  
  - Phonetic name (Double Metaphone / Soundex)  
  - First meaningful name token + state/city/PIN if present  
  - Character n-gram TF-IDF + ANN (Faiss / hnswlib) on name (+ address)  
  - Separate India / US / France address parsers (rules, not external APIs)  
- [ ] Measure **blocking recall** on a train holdout (must be high) and **avg candidates per S1** (keep small)  
- [ ] Baseline matcher: Jaccard / TF-IDF cosine / Levenshtein features + logistic or XGBoost  
- [ ] Tune threshold for **F₀.₅**; bias toward precision  
- [ ] First valid `matching_results.tsv` + `candidate_pairs.tsv` → validate → submit  

#### Day 2 — Improve model + France generalization

- [ ] Richer features: token Jaccard, partial ratios, digit/PIN overlap, sorted-token cosine, length ratios, country match flag  
- [ ] Optional: multilingual embeddings (MIT/Apache, ≤8B) for name+address; still block first  
- [ ] Calibrate for singletons (don’t over-predict)  
- [ ] Error analysis: false merges vs misses; fix blocking holes vs threshold  
- [ ] Scale pipeline to full test (memory, chunking, Spark/Dask if needed)  
- [ ] Mid-hackathon push for Top-500 extra $100 if credits help larger runs  

#### Day 3 — Harden + package

- [ ] Freeze threshold; re-run full test inference  
- [ ] Validate with `utils/validate_submission.py` (optionally `--check-ids` if memory allows)  
- [ ] Fill `Documentation_template.md` (blocking, features, model, F₀.₅ on validation)  
- [ ] Build zip with reproducible `code/business_entity_resolution/`  
- [ ] Keep version history of each portal submission (max 5/day)  
- [ ] Prefer last **safe high-precision** submission over last-minute risky changes  

### Suggested stack (practical for 72h)

- Python, pandas/polars, numpy  
- Blocking: rapidfuzz / jellyfish + sklearn TF-IDF + faiss/hnswlib  
- Model: **XGBoost / LightGBM** on pair features (strong baseline; matches AWS demo spirit)  
- Optional later: sentence-transformers-style embedding model if license OK and size ≤8B  
- Parallelism: joblib / multiprocessing; process by country shards  

### What NOT to waste time on

- Deploying SageMaker endpoints  
- End-to-end LLMs without blocking (won’t scale; license/size risk)  
- Perfecting France geocoding via Google Maps (disallowed)  
- Overfitting public leaderboard with noisy merges  

---

## 6. Local validation recipe

Hold out e.g. 10–20% of Source-1 train entities (stratify by country / has-match). For each S1:

```
P = |predicted ∩ true| / |predicted|   (1.0 if both empty)
R = |predicted ∩ true| / |true|       (1.0 if both empty)
F0.5 = 1.25PR / (0.25P + R)           (handle zeros carefully)
```

Average F₀.₅ over all holdout S1 IDs. Also track:

- Blocking recall (fraction of true matches present in candidates)  
- Avg / p95 candidates per S1  
- Singleton accuracy  

---

## 7. Submission checklist

**Every leaderboard upload**

- [ ] Tab-separated, exact headers  
- [ ] One row per test S1 entity  
- [ ] Only S2-/S3- IDs that exist in test  
- [ ] No duplicate IDs in a list; no duplicate S1 rows  
- [ ] `python3 utils/validate_submission.py ...` → `PASS`  
- [ ] Under 5 submissions that day  

**Final package**

- [ ] `output/matching_results.tsv`  
- [ ] `output/candidate_pairs.tsv` (superset of matches; last stage before model)  
- [ ] Runnable code + `requirements.txt` + reproduce README  
- [ ] Filled `Documentation_template.md`  

---

## 8. Key links

| Resource | URL / location |
| --- | --- |
| Problem (local) | `README.md`, `problem statement.pdf` |
| Guidelines (local) | `guidelines.pdf` |
| AWS Builder prep guide | https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo |
| Builder Center signup | https://builder.aws.com |
| Student Rewards | https://aws.amazon.com/builder |
| Query form (during hackathon) | linked in `guidelines.pdf` / emails |
| Problem explainer video | linked in `problem statement.pdf` |
| Unstop support | support@unstop.com |

---

## 9. Limited AWS credits — how we work

**Default: $0 AWS spend.** Data is already on your Mac under `student_resource/`. The scaffolded pipeline is **CPU-only** (sklearn HistGradientBoosting + rapidfuzz + inverted-index blocking). No GPU, no SageMaker endpoints, no Training Jobs unless you explicitly need them.

| Priority | Where to run | Credits |
| --- | --- | --- |
| 1 (default) | Local laptop / desktop | **$0** |
| 2 (backup) | One SageMaker Notebook `ml.t3.medium`, stop when idle | Low |
| 3 (avoid) | GPU / multi-instance / endpoints | Burns credits fast — skip |

**Credit rules for this team**

- Develop with `--sample 5000` / `20000` until metrics look good  
- Full train/infer overnight on the laptop if possible  
- If AWS is required: **one** `ml.t3.medium` notebook only; store data in S3; stop notebook every break  
- Set a **billing alert** immediately if you open an AWS account  
- Prefer raising `max_candidates` carefully over spinning bigger machines  
- Extra $100 (top 500 at 48h) is a bonus — do not plan the budget around it  

Scaffolded code lives at:

`code/business_entity_resolution/`

Quick start:

```bash
cd student_resource
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
bash code/business_entity_resolution/scripts/smoke_test.sh
```

---

## 10. Immediate next actions for you (right now)

1. Confirm dataset is complete (you already have train + test under `dataset/`).  
2. **Do not** open a SageMaker notebook yet — install deps and run the local smoke test first.  
3. Create Builder Center profile only if registration still needs it; unlock credits later as backup.  
4. Iterate: sample train → check val F₀.₅ → grow sample → full infer → validate → portal upload.  
5. Keep submissions conservative (precision-first) and under 5/day.