# Matching procedure, results, and what to improve

This file describes the pipeline as implemented. It does not restart training or test scoring.

## What the task is

Source 1 is the reference. For every Source 1 business, list the Source 2 and Source 3 records that are the same real business, or leave the list empty. The portal score is macro F₀.₅: precision counts twice as much as recall. A wrong link on a business that should have no matches scores 0 for that business.

Only the provided name, address, and country are used. No web lookup, geocoding, or business registry.

## 1–4. Batching, and the criteria

Yes. Each Source 1 row is not compared with all of Source 2 and Source 3.

Batching happens before matching. Records that share a blocking key are the only ones compared. Keys are built inside one country (the country string is kept as-is, so France is included even though it is not in the training countries).

A record emits these keys:

- First two significant name words, sorted so word order does not matter. If there is a third word, also the pair of the first and the third.
- One PIN or ZIP of 5 or more digits.
- The two longest distinctive address words.
- One long locality word (8 or more letters).
- House number plus that locality word.

Keys that hit more than 25 businesses are dropped. That is what stops a common name such as “My Builders” from linking Kolkata to Mumbai. Each Source 1 row then keeps at most 12 candidates. Address and house-number keys rank above a plain name key.

Text is normalized first: lower case, `&` becomes “and”, legal endings such as Ltd, Pvt, Inc, Corp, LLC are expanded and then removed, and address shorts such as Rd, St, Ave are expanded. There is no dictionary of the form `LLT = company` or `TM = company`.

## 5. Where the code lives

This is not a Jupyter notebook. The steps are Python scripts:

- `code/business_entity_resolution/src/ber/normalize.py` — text cleanup
- `code/business_entity_resolution/src/rank_match.py` — blocking, classifier, held-out score, test run
- `code/business_entity_resolution/src/infer_fast.py` — the earlier string-score submissions
- `code/business_entity_resolution/matching_procedure.ipynb` — the same steps written in order, not executed against the full data

## 6. Models and scores

| Approach | What it is | Score | Where |
| --- | --- | --- | --- |
| String score, cutoff 0.84 | RapidFuzz name ratio, token overlap, Jaro-Winkler. No trained model. | 0.418798 | Public leaderboard |
| Stricter string score, cutoff 0.90 | Same idea, harder cutoff | 0.416 | Public leaderboard |
| HistGradientBoosting, early training | Trained with the true match already placed in the pool | about 0.89 local | Not submitted. The 0.89 number is inflated. |
| HistGradientBoosting, current | Name closeness, sorted-address closeness, address-word containment, missing-address flag. Cutoff 0.72 | 0.678 local held-out | Not submitted yet. File still being written to `output_v3/`. |
| Hand rule on the same candidates | Same address, or close name and not a different city | 0.582 local held-out | Rejected in favor of the classifier |

The current model sees only those four signals. It was fit on about 3,900 Source 1 businesses and scored on about 1,650 others. Blocking found 65% of the true pairs in that slice (about 6.6 candidates per Source 1 business).

## 7. Abbreviation metadata

No phrase dictionary for swaps such as `LLT` / `TM` / `DBA`. Only the small legal-suffix and street-abbreviation lists above, plus stopwords that are not used as blocking keys (`trading`, `private`, `road`, `nagar`, and similar).

## 8–9. What matters, and what would raise the score

- Do not upload `output_v3/matching_results.tsv` until the run prints that it is done. A partial file will be rejected or scored unfairly.
- The 0.42 files failed mainly by linking the same common name across different cities, and by never seeing matches that share an address but not the name (trade name, domain, Hindi spelling).
- Raising the cutoff from 0.84 to 0.90 made the public score worse. The next gain is not a higher cutoff.
- Blocking recall is 65%. Any true pair that never enters the list of 12 cannot be saved by the classifier. That is the main ceiling under the leader’s 0.99.
- Useful next steps, still using only the given files: typo-tolerant keys (character groups) that are still deleted when too common; a small abbreviation map of the kind in question 7; train on more than 5,500 Source 1 rows; keep the candidate list small, because the final review prefers fewer candidates per Source 1 business.
- France is only in the test set. The pipeline does not filter to US and India.
- Empty predictions are required for businesses with no match. Predicting a link for one of those scores 0 on that row.
