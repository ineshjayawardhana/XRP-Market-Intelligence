# XRP research classification, Silver/Gold data and batch review

## Start the review interface

Run these commands from the project folder:

```powershell
.\.venv\Scripts\python.exe research_workflow.py
.\.venv\Scripts\python.exe -m streamlit run review_research_app.py --server.port 8502 --server.address 127.0.0.1
```

Open **http://127.0.0.1:8502**. Port 8502 keeps this separate from the existing live
dashboard on port 8501. The first command is offline: it rebuilds research files
and reports without model fitting or network requests. The second starts only the
local review interface. No existing dashboard or trained model is changed.

The current completed pass assessed all **400 records**, produced **215 Silver
labels**, retained **8 existing Gold annotations** and all **3 SKIP decisions**, and
left **174 pending records REVIEW_REQUIRED**. Gold and Silver are disjoint.

| Category | Silver provisional | Existing Gold human |
| --- | ---: | ---: |
| Real_Adoption | 10 | 0 |
| Hype_Speculation | 29 | 1 |
| Negative_FUD | 13 | 1 |
| Neutral_Market_News | 163 | 6 |

These are annotation counts, not measured model accuracy. The Gold rows are seven
legacy manual annotations and one previously saved human acceptance. No new human
annotations were created by the automated pass. Legacy reviewer/timestamp blanks
remain blank; their history is not invented.

## Review multiple records quickly

1. Begin with **Recommended sample**, or filter All pending/Uncertain only.
2. Filter by category, publisher, confidence band and uncertainty. Grouped count
   tables use the same filters. Choose 10, 25, 50 or 100 rows per page.
3. Open the evidence panel to see the full headline, article link, reasoning,
   excerpt, its origin and retrieval time. Excerpts are not independent fact checks.
4. Select the records you reviewed. Choose per-row accept/correct/reject/defer,
   or choose one default action and corrected category for the selected rows.
5. Enter your reviewer name and press **Save selected decisions**.

Select-all applies only to the currently displayed batch and starts unchecked.
Blank or deferred actions never create Gold labels. REVIEW_REQUIRED cannot be
accepted; choose a human category after resolving it or reject/defer. Rejection
does not assign a category. Existing human and SKIP/rejected decisions are read-only.

The complete batch is validated before any ledger write. A stale prediction,
changed article/timestamp, duplicate submitted ID or invalid action prevents partial
saves. Human decisions append to `data/multisource/xrp_human_reviews.csv` with
reviewer and UTC timestamp, backed up first. Source CSV labels are not overwritten.
The app then refreshes the Gold/Silver files and reports. If refresh fails after a
successful save, the app states that the decisions were saved and offers Refresh;
the authoritative ledger remains intact.

The original root datasets and review files remain untouched. Other existing tools
already read the multisource ledger. Use **Refresh datasets and readiness** after
reviewing through those tools. The default dataset directory can be overridden
with the `XRP_RESEARCH_DATA_DIR` environment variable for another workspace.

## Files under data/multisource/research/

| File | Meaning |
| --- | --- |
| all_classifications.csv | Assessment of all 400 rows, including protected human/SKIP rows |
| silver_dataset.csv | Unique AI_PROVISIONAL category assignments with provenance and reasoning |
| gold_dataset.csv | Human content annotations only, retaining their original/review provenance |
| review_queue.csv | Every still-unreviewed record, including provisional and uncertain suggestions |
| verification_sample.csv | Recommended first verification sample and temporal strata |
| readiness_report.json / .md | Counts, quality flags, quota gaps and training readiness |
| additional_context.json | Cached public article attempts, failures, timestamps and excerpts |
| context_access.sqlite3 | Separate host robots/cooldown state for context requests |
| training_prepared/ | Human-only chronological candidate partitions and sufficiency checks |

The older `xrp_auto_labeled.csv` and batch reports are preserved as previous
assessments; use the **research** outputs for the improved classification.
Repeated builds back up existing research CSVs. The original multisource inputs
were also snapshotted before this work. The collector's database is not modified.

## Evidence and classification method

`research_content_rubric_v2` is an explainable offline content rubric, not a newly
trained language model or a live LLM API call. Matching existing AI assessments are
reused when appropriate. Strong disagreement is flagged for review. The rubric:

- Requires substantive support for an operational product/integration. Plans,
  voting proposals, old launches on different chains and a new webpage do not
  prove the announced deployment is live.
- Distinguishes reported security/legal incidents from price declines, derivatives
  measurements and scheduled testnet resets.
- Distinguishes observed market statistics from attributed price targets and
  future-return speculation. Forecasts embedded in balanced technical/valuation
  discussions can remain uncertain.
- Leaves promotions, mixed roundups, incidental multi-asset outlooks and unclear
  claims unresolved. Sensational wording is flagged, not declared false.
- Detects excerpts that simply repeat the headline. They are treated as headline-only
  evidence. Clear headline-only content can support a provisional category; it never
  establishes factual verification or completed deployment.

Confidence values are heuristic ranking scores, **not calibrated probabilities**,
truth scores or predictions of future XRP prices. Every automatic label remains
AI_PROVISIONAL. HUMAN_VERIFIED denotes a human's content annotation, not independent
article fact checking. All records retain `article_verified=False`.

Optional bounded context collection:

```powershell
.\.venv\Scripts\python.exe research_workflow.py --fetch-context 20 --max-requests 35
```

Requests use the existing public client: robots checks, ordinary redirects,
host-wide cooldowns, Retry-After, declared crawl delays and request limits. The
extractor checks article headings and semantic/standard public content containers,
and refuses declared restricted-reading pages. It does not log in, bypass paywalls,
decode aggregator links or treat a 200 response as verification. New excerpts are
limited to 180 words, with retrieval timestamps and hashes.

Cached failures prevent automatic repeated requests. After an extraction
improvement, `--retry-unavailable-context` can retry only earlier extraction
failures; access blocks are not eligible, and robots/cooldowns still apply. Prior
attempts are backed up. The completed work obtained six usable additional public
excerpts. Remaining access failures are included in the readiness JSON.

## Representative verification plan

The default goal is **25 distinct verified examples per category**, configurable
with `--target-per-class 20` through `30`. Existing eligible distinct Gold labels
count toward the goal. Candidates are spread across publication-time quartiles,
publishers and confidence bands where available. Missing publication dates fall
back to first-available time. An additional 12 uncertain records are spread over
time for an uncertainty audit.

The current recommended set is **78 records**: 10 adoption, 24 speculation,
13 negative, 19 neutral and 12 uncertainty-audit records. Even if every category
suggestion is confirmed, adoption remains 15 short of 25 and negative remains
11 short. Human corrections may change these numbers. The workflow does not
manufacture candidates or approvals to meet quotas. More genuinely relevant
articles/context and human annotation are needed for the underrepresented classes.

Publication dates are used for **sample coverage only**, not to backdate training
or backtesting availability. An archive sample is chronologically representative
of reported publication periods but cannot prove the project possessed that news
at those earlier dates.

## Quality and model readiness

The report checks detected duplicate groups, conflicting AI/human labels,
conflicting duplicate labels, missing publication dates, invalid availability,
source imbalance, weak excerpts, sensational framing and possible unit confusion.
The corpus retains one legacy duplicate pair and 52 missing publication dates.
The largest publisher supplies 31.75% of records. Flags indicate review needs,
not independently established misinformation.

There are 396 distinct non-SKIP candidates with valid availability; that number
includes unresolved annotation candidates and is **not** a training-set size.
Only 8 Gold labels exist, and current chronological embargoes leave 3 human
training rows, 0 validation rows and 2 test rows. Both TF-IDF and DistilBERT
supervised training are **not ready**. Dependencies are reported but no transformer
weights are downloaded.

The preserved minimum remains 120 distinct human records **after** duplicate checks
and embargo, at least 20 per class, and each chronological partition must retain
at least ten human examples and all four classes. Silver cannot satisfy those
requirements. It remains an unassigned weak-label pool and never enters evaluation
as verified ground truth. No accuracy or future-price probability is generated.

First-available/project-observed timestamps drive preparation; original timestamps
remain unchanged. Most newly collected headlines were observed on one day, so more
human labels alone will not repair the limited temporal coverage and 24-hour gaps.
Collect across additional observation dates. Any stronger historical availability
claim needs separate evidence. Models use headlines only; later-fetched article
text, AI confidence and annotation metadata are not features. Independent/blind
annotation is preferable for a future held-out academic benchmark.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests include the existing collection/training safeguards plus evidence-rubric
regressions, Silver/Gold disjointness, preserved human/SKIP decisions, atomic batch
saves, stale snapshots, context restrictions/caching, temporal/source sampling,
and Streamlit simulation of explicit multi-row saves and empty/read-only queues.
They write temporary fixtures only; no actual human decisions or model fits occur.
