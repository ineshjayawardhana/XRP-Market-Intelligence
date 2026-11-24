# Automatic XRP labeling and safe training preparation

## Run the batch

```powershell
.\.venv\Scripts\python.exe ai_labeling.py batch
```

This processes every existing suggestion, plus new blank rows in the real dataset,
without per-headline prompts. Existing CSVs receive timestamped, hash-checked
backups in `data/backups/` before outputs are replaced. The original dataset,
AI suggestions and human ledger remain unchanged.

For new articles, also fetch or retry missing context in the same batch:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py batch --refresh-missing-context
```

This uses the existing requests library with bounded timeouts. Blocked or missing
articles are recorded as unavailable. It does not bypass access restrictions.
Existing excerpts are reused; retrieved text is never represented as fact-checked.

The classifier uses matching saved AI assessments with per-record reasoning and
excerpt checks. For new records without saved assessments, it uses narrow,
explainable evidence rules. There is **no live LLM API call**, model download,
paid service, or requirement to submit new proposal files. This is a conservative
offline classifier, not a newly trained or independently validated language model.

Article excerpts must support rule-based assignments. Missing usable context,
scores below the heuristic threshold, unclear or mixed themes, and conflicting
assessments result in REVIEW_REQUIRED. Detected syndicated duplicates and conflicts
with protected human decisions are also flagged. Human decisions always take
precedence. The default `--min-confidence` is 0.75: a heuristic threshold, not a
calibrated probability, truth-verification score, or measured accuracy.

## Generated files

| File in data/ | Purpose |
| --- | --- |
| xrp_auto_labeled.csv | All predictions, scores, per-row reasoning, sources/URLs, timestamps, provenance and context evidence |
| xrp_batch_review.csv | Uncertain, conflicting or duplicate records requiring attention |
| xrp_batch_report.md | Readable counts and flagged-record list |
| xrp_batch_summary.json | Machine-readable counts and input/output hashes |
| xrp_batch_decisions_template.csv | Editable multi-record human review sheet |

All automatic outputs have `label_status=AI_PROVISIONAL`,
`verification_status=NOT_FACT_CHECKED`, and `article_verified=False`.
`timestamp` is the source availability/first-seen time, `classified_at` is the
processing time, and `context_retrieved_at` describes the page snapshot.
Stored timestamps are UTC. The excerpt and its SHA-256 are retained per record.
Page extraction may include sidebars or incomplete text; it does not independently
verify article facts or original publication times.

The existing 43 suggestions produced **23 provisional categories**: 2 adoption,
1 speculation, 0 negative, and 20 neutral. **20 remain REVIEW_REQUIRED**.
The batch used 37 available excerpts. All six unavailable-context records remain
uncertain. Two earlier headline-only adoption suggestions and one headline-only
negative suggestion were moved to REVIEW_REQUIRED in the new output. The original
suggestions are preserved.

## Review multiple records together

Provisional labels can remain provisional; you do not need to review every record
to run the workflow. To record human decisions for selected records, copy
`data/xrp_batch_decisions_template.csv` to `data/my_batch_decisions.csv` and edit
any number of rows:

| action | human_label | Result |
| --- | --- | --- |
| accept | blank | Accept the displayed category |
| correct | One of the four categories | Record the chosen human category |
| reject | blank | Reject without assigning a category |
| defer or blank | blank | Leave unresolved |

Keep `id`, `headline`, `url` and `predicted_label` unchanged. You can remove rows
you do not wish to review. REVIEW_REQUIRED cannot be accepted as a category.

```powershell
.\.venv\Scripts\python.exe ai_labeling.py batch-review --decisions-file data/my_batch_decisions.csv --reviewer "Inesh"
```

The importer validates the entire batch before saving it atomically to the human
ledger. It refuses stale records, invalid categories, repeated decisions and
changes to existing human labels or SKIP rows. A blank action is never approval.
Rerunning a committed decision file is refused to protect human decisions.

The source dataset still does not change. The original `export-reviewed` command
remains available to promote human decisions into blank source labels. Preparation
also reads valid human decisions directly from the ledger, so export is optional.
Run `batch` again to refresh reports after recording decisions. Copy the template
before editing it; the generated template is replaced on each batch run.
Close independent labelers/collectors before writes; the workflow lock coordinates
only the new commands.

## Prepare training data without retraining

```powershell
.\.venv\Scripts\python.exe prepare_real_training.py
```

This creates `data/training_prepared/` containing separate human train, validation
and held-out test CSVs, a provisional training CSV, an exclusion audit and a
readiness manifest. It does not fit or save a model.

- Existing manual labels are legacy human content annotations. Saved human
  decisions need exact ID/headline/URL matches, reviewer and UTC review timestamp.
  Blank, SKIP, rejected and explicitly AI-provisional records never become human
  ground truth. AI-assisted human decisions retain their distinct origin.
- HUMAN_VERIFIED means a human classified the content, not that the article was
  fact-checked. Independent annotation is preferable for an academic test set.
- Deduplication groups IDs, canonical URLs without tracking parameters,
  publisher-stripped and near-identical titles, and substantial excerpt overlap.
  Conflicting human duplicate labels quarantine the group. Human annotations take
  priority over provisional copies. No detected group enters multiple partitions;
  paraphrase detection remains heuristic.
- Human records split approximately 70/15/15 by UTC availability time with a
  24-hour embargo before validation and test. Invalid, naive and future timestamps
  are excluded. GDELT first-seen is not verified original publication time.
- Validation and test are human-only. Provisional training is excluded by default.
  `--include-provisional-training` exports a separate weak-label training file
  restricted to records before the validation boundary and embargo. It never joins
  human-only partitions or the standard trainer.
- Once sufficient data exists, test IDs and content signatures are frozen in
  `test_reservation.json`. New records on/after that boundary cannot enter training.
  Changed or missing frozen records block readiness. Before sufficiency, test CSVs
  are tentative reservations, not a valid accuracy benchmark.
- Input/output hashes detect changed, removed or newly created inputs and edited
  prepared partitions. The standard trainer prepares and validates them again
  before fitting.

`train_real_model.py` now uses this preparation and refuses to fit if readiness is
false. It uses **headlines only** as features, with no future prices, excerpts, AI
scores or annotation metadata. Existing training helpers and dashboard remain.

There are currently **8 human annotations**: 6 neutral, 1 speculation, 1 negative
and 0 adoption, including the saved human acceptance. The original source CSV
still contains seven category labels, three SKIP decisions and 43 blank labels.
After the embargo, tentative partitions contain 3 training, 0 validation and
2 test rows, so readiness is false.

Before training, collect enough human annotations across more dates to retain
**at least 120 distinct human records after duplicate checks and embargo**, with
**at least 20 per class** overall. Each chronological partition must retain at
least 10 human records and all four classes. AI-provisional labels cannot satisfy
these requirements. No model accuracy is calculated from this batch.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary fixtures, never real dataset edits or model fitting. Coverage
includes provisional provenance, byte-preserved inputs, human/SKIP protection,
missing/stale context, conflicts, multi-record review, all-or-nothing validation,
AI exclusion from evaluation, duplicate grouping, temporal embargo, frozen test
reservations, tampered files and refusing model fitting when data is insufficient.
