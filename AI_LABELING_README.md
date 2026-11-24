# AI-assisted XRP labeling

For automatic classification and multi-record review, use the new
[batch workflow](BATCH_LABELING_README.md): `python ai_labeling.py batch`.
It creates AI_PROVISIONAL outputs and safe chronological training preparation
without per-headline prompts. The original interactive commands below remain
available. The `generate` command still manages saved suggestions; `batch` can
also process new articles without supplied proposals.

The initial 43 blank rows have individual AI-assisted suggestions in
`data/xrp_ai_suggestions.csv`. Your original `data/xrp_real_labeled.csv` remains
unchanged. All existing category labels and `SKIP` decisions are preserved.
The first backup is `data/backups/xrp_real_labeled_20261115T0717376284361Z.csv`.
Its SHA-256 matches the original dataset:
`96a459af434ffeb5bd1f31a8a23f4b32349f0e228f7bbefb1c1a051baeb15ca6`.

## Review quickly

From this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py review --reviewer "Your name"
```

Read the headline, article link, proposed category and reasoning, then press a key
followed by Enter:

| Key | Action |
| --- | --- |
| `a` | Accept a proposed category |
| `1` | Set Real_Adoption |
| `2` | Set Hype_Speculation |
| `3` | Set Negative_FUD |
| `4` | Set Neutral_Market_News |
| `r` | Reject the suggestion without labeling the article |
| `d` | Defer until a later review session |
| `q` | Quit and resume later |

`REVIEW_REQUIRED` cannot be accepted as a category. Choose `1–4` after resolving
the uncertainty or reject/defer. Invalid answers keep the same article on screen.
`--count 10` limits a session to ten displayed articles. Every decision saves
immediately. Review resumes with pending entries. Rejected suggestions are excluded
from the normal queue and never exported. The human decision ledger is
`data/xrp_human_reviews.csv`; the tool creates it only when you make a decision.
It records the reviewer, UTC timestamp, human label and original suggestion.
The interactive tool has no accept-all action. The batch workflow supports
multiple explicit human decisions in one CSV import.

After reviewing, explicitly promote human decisions into blank dataset labels:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py export-reviewed
```

This command backs up the dataset, checks exact ID/headline/URL matches, and fills
only blank labels from accepted/corrected human decisions. It never changes an
existing category or `SKIP`. It adds `label_origin=human_reviewed_ai`,
`human_reviewed_by` and `human_reviewed_at` to exported rows. Repeating it has no
effect on already labeled rows. AI labels remain in the suggestion file. Rejection
does not set `SKIP` in the dataset. Your original manual labeling script remains
available to handle those articles independently.

## Categories and uncertainty

- Real_Adoption: a specific product, deployment, integration or operational use of
  XRP, XRPL or RLUSD. RLUSD integration does not establish use of XRP tokens or
  future XRP price gains.
- Hype_Speculation: price-centered speculation, sensational targets, rumors or
  claims of assured returns. A conditional technical level alone is insufficient.
- Negative_FUD: negative incident, legal/regulatory risk, outage or security issue.
  A price decline alone is not sufficient.
- Neutral_Market_News: ordinary market reporting, ETF/flow data, balanced analysis
  or factual explanatory coverage.
- REVIEW_REQUIRED: unresolved category boundaries, unrelated promotion, or mixed
  adoption, price and risk themes. Never train this as a fifth class.

`confidence_estimate` is a subjective heuristic score from 0 to 1, **not a
calibrated probability**, measured accuracy or factual verification. For
REVIEW_REQUIRED it reflects low confidence in assigning any single category.
All initial entries have `review_status=pending` and `needs_manual_review=True`.
The latter becomes False after a human decision. Even a high score needs review.

## Article context and reproducibility

`data/xrp_article_context.json` records an access attempt for each of the 43
articles, including URL, retrieval timestamp, status and a short candidate excerpt
when available. Relevant excerpts were obtained for 37; six were unavailable
(blocked, missing, or no usable paragraphs). `retrieved_unverified` means text was
available, not that the facts or original publication date were verified.
Every suggestion has `article_verified=False`; the reasoning records headline-only
limitations where applicable. The URL and per-row context status travel with the
suggestion. The excerpts were used to distinguish conditional analysis from hype
and concrete integrations from general adoption discussion.

Extraction is deliberately limited: it may include related-story/sidebar text,
duplicate paragraphs or decoding artifacts. The initial assessment ignored
obvious unrelated sidebar price snapshots. Fetches reflect the current page,
which may differ from its original version. Open the linked page to resolve
uncertainties. No future market outcomes were used to label headlines.

The initial AI assessments, bound to exact ID/headline/URL values, are saved in
`data/xrp_ai_proposals.json`. They can reproduce the initial CSV:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py generate
```

Reruns preserve previous suggestions and human decisions, including historical
suggestions already exported. Human decisions are authoritative if a process stops
between saving the ledger and refreshing CSV review statuses. Run `generate` to
resynchronize statuses. All CSV writes use an atomic replacement. A workflow lock
prevents two instances from overwriting each other's work. Close the original
manual labeler/collector and any spreadsheet editor before exporting. The lock
does not control those independent programs. If a workflow crashes, remove
`data/.ai_labeling.lock` only after confirming that no workflow is running.
An ID with a recorded human decision cannot receive another decision through this
tool. If the underlying article changes, retain the original audit record and use
a new ID for the changed item. Stale decisions never export to changed content.

## Additional news

Collect and merge new articles using your existing scripts, then run:

```powershell
.\.venv\Scripts\python.exe ai_labeling.py fetch-context
.\.venv\Scripts\python.exe ai_labeling.py generate --proposals data/my_new_proposals.json
```

`fetch-context` attempts only blank rows with bounded HTTP timeouts and four
workers. It replaces the current context snapshot; archive it first if you need
to preserve earlier retrieval evidence. It does not bypass blocked pages.

This implementation has no external LLM dependency, API key, cost, or hidden
model call. The initial 43 assessments were authored during this task. For new
articles, provide another AI assessment JSON list using the initial proposal file
as the schema. Each entry needs `id`, `headline`, `url`, `suggested_label`,
`confidence_estimate`, `reasoning`, `context_status`, and `suggestion_method`.
Without a matching proposal, a new article safely becomes REVIEW_REQUIRED with
score 0 and no claim of article verification. Fetching context alone does not
generate an AI assessment. Existing matching suggestions are retained to protect
the review record; use a separate `--data-dir` if you want to compare a new batch
of assessments with old ones.

## Academic evaluation safeguards

The trainer reads only `data/xrp_real_labeled.csv`, never the suggestions or
proposal files. Suggestions intentionally have no `label` column. When provenance
is present, explicitly AI-origin and unknown-origin labels are excluded. Exported
AI-assisted labels require a human reviewer and timestamp. Blank origin values
retain the existing project's assumption that legacy dataset labels are manual;
never paste AI labels into that file without provenance.

Human-reviewed AI-assisted labels are eligible only after explicit export.
Human acceptance can introduce anchoring bias. For a stronger academic test set,
use independently annotated, preferably blind-reviewed articles and report how
many labels were AI-assisted. Article factual verification and content annotation
are different tasks. The original minimum of 120 distinct articles and at least
20 per class, chronological splits, deduplication and 24-hour embargo remain in
place. The initial seven category-labeled articles cannot produce a valid model
test accuracy. No accuracy is calculated from these suggestions.

The two “Will XRP Hit $4 By 2028?” stories appear syndicated across publishers.
Inspect cross-publisher duplicates before evaluation; the trainer's normalized
headline deduplication does not catch every syndicated variation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary CSVs and simulated keyboard input. They cover byte-identical
backups, blank-only generation/export, accepted/corrected/rejected/deferred
decisions, invalid input, resume, stale content, reruns, duplicate IDs, invalid
scores, locking, unavailable articles, AI exclusion and minimum-data protection.
They never create human review decisions in your real dataset. The dashboard and
existing models are unchanged.
