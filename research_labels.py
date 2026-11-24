"""Versioned, conservative content rubric for provisional XRP research labels.

This is an explainable offline rule system, not a calibrated or newly trained LLM.
It distinguishes an actual deployment from a proposal and a reported incident from
a bearish price chart. Extracted text is evidence of coverage, never factual proof.
"""
from __future__ import annotations

import re
import math

from ai_labeling import LABELS, REVIEW, matching

VERSION = 'research_content_rubric_v2'


def clean_excerpt(headline, text):
    text = re.sub(r'\s+', ' ', text or '').strip()
    text = re.split(r'\bThe post\b|\bRelated Reading\s*:', text, maxsplit=1, flags=re.I)[0].strip()
    if re.search(r'^read the latest update|what the latest source material shows', text, re.I):
        return ''
    title_words = set(re.findall(r'[a-z0-9]+', headline.casefold()))
    words = re.findall(r'[a-z0-9]+', text.casefold())
    extra = set(words) - title_words
    if len(words) < 8 or (len(extra) < 4 and len(words) <= len(title_words) + 8):
        return ''
    return ' '.join(text.split()[:180])


def assess(headline, excerpt):
    title, body = headline.casefold(), excerpt.casefold()
    flags = []
    if re.search(r'bombshell|face.melting|moonshot|bloodbath|astronomical|military.grade|guaranteed|\binsane\b', title):
        flags.append('sensational_framing_not_fact_checked')
    if '\ufffd' in headline or '\ufffd' in excerpt:
        flags.append('source_text_encoding_artifact')
    if not re.search(r'\b(xrp|xrpl|rlusd|ripple)\b', title):
        return REVIEW, .2, 'No clear XRP ecosystem subject in the headline.', flags
    if re.search(r'presale|alphapepe|remittix|ftmining|daily passive income|cloud mining', title):
        return REVIEW, .35, 'Unrelated token or income promotion is mixed with XRP coverage.', flags + ['unrelated_promotion']
    if re.search(r'hodler.s digest|weekly (wrap|roundup)|important ripple news and xrp price|announcements.*day 0|and more$', title):
        return REVIEW, .4, 'A roundup combines several events; one content category would conceal mixed evidence.', flags + ['mixed_topics']
    if re.search(r'updates outlook|outlines path forward.*(bitcoin|ethereum)|predicts bitcoin|calls xrp.*fool|saylor.s bitcoin', title):
        return REVIEW, .4, 'XRP is embedded in another asset outlook or unrelated commentary.', flags + ['incidental_or_multi_asset_outlook']
    if re.search(r'nvidia|jensen huang', title):
        return REVIEW, .4, 'Cross-industry headline links an external event to XRP; the relationship requires checking.', flags + ['cross_topic_claim']

    available = bool(excerpt)
    evidence = .84 if available else .72
    if re.search(r'pre.lawsuit levels|buy side explodes.*liquidity|liquidity.*heavier than sells', title):
        return 'Neutral_Market_News', evidence, 'This reports an observed demand/liquidity measurement, not a new lawsuit or future return.', flags
    if available and re.search(r'testnet|devnet', title) and re.search(r'reset|wipe', title):
        if re.search(r'scheduled|testnet|devnet', body):
            return 'Neutral_Market_News', .85, 'The excerpt describes scheduled test-network maintenance, not a production loss or outage.', flags

    proposal = bool(re.search(r'proposal|proposes|roadmap|plans to|prepar(?:es|ing) to|not yet available|subject to regulatory approval|nears launch', title + ' ' + body[:450]))
    product = bool(re.search(r'launch(?:es|ed|ing)?|rolls? out|rolled out|goes live|go live|integrat(?:es|ed|ion)|custody|settlement support|payment options|credit card|starter kit|new use case|partners? with|partnership', title))
    deployment = bool(re.search(r'launch(?:es|ed)|rolled out|rolls out|has (?:launched|rolled)|is (?:now )?available|now available|has integrated|integrates|goes live|has partnered|pays.*back in xrp|settlement support', body))
    if re.search(r'launches|rolls out|goes live|integrates|adds.*custody', title) and re.search(r'fund|tokenized|custody|settlement|payment|wallet|credit card|developer', body):
        deployment = deployment or not re.search(r'preparing to|plans to begin|not yet available|subject to regulatory approval', body[:250])
    price_mix = bool(re.search(r'price (?:impact|prediction|target)|push.*price|price.*\$|but xrp|price struggles', title))
    if product and re.search(r'webpage|website|gears up|nears launch|plans to|to launch', title):
        return REVIEW, .4, 'A webpage, announced plan or upcoming rollout does not prove service availability.', flags + ['proposed_not_deployed']
    if product and re.search(r'launching', title) and re.search(r'preparing|plans to begin|begin testing', body[:400]):
        return REVIEW, .4, 'The new launch is still planned; an older launch elsewhere in the excerpt does not establish this deployment.', flags + ['proposed_not_deployed']
    if product and re.search(r'predict|forecast', title):
        return REVIEW, .45, 'The headline mixes a partnership/product announcement with a speculative outlook.', flags + ['mixed_adoption_and_price']
    if product and price_mix:
        return REVIEW, .45, 'A product/adoption theme is combined with a price argument.', flags + ['mixed_adoption_and_price']
    if available and re.search(r'new xrp trust|xrp futures|exchange.traded fund', title):
        return 'Neutral_Market_News', .8, 'This concerns an investment product or derivatives listing; it does not by itself establish XRPL operational adoption.', flags
    if product and available and deployment and not re.search(r'proposal|plans to|preparing to|nears launch', title) and not re.search(r'not yet available|subject to regulatory approval', body[:250]):
        return 'Real_Adoption', .86, 'The headline names a concrete service or integration, and the excerpt describes its launch or availability.', flags
    if product and proposal:
        # An explicitly running beta is a pilot deployment, not a production launch.
        if available and re.search(r'kicked off beta testing|began (?:a )?pilot|pilot is live', body):
            return 'Real_Adoption', .78, 'The excerpt describes an operating beta/pilot; this label does not assert general public availability.', flags + ['pilot_not_general_availability']
        return REVIEW, .45, 'A proposed, planned or not-yet-available service does not establish a completed deployment.', flags + ['proposed_not_deployed']
    if product and not available:
        return REVIEW, .4, 'Product wording lacks a substantive excerpt confirming the status of the service.', flags + ['deployment_context_missing']

    if proposal and available and re.search(r'validator|amendment|protocol|security|lending', title):
        return 'Neutral_Market_News', .8, 'This is a documented proposal or governance update; it is not treated as an activated deployment.', flags
    if available and re.search(r'bug fix|hotfix|patch|security upgrade|releases? xrpld|version.*release', title):
        if re.search(r'vulnerab|unlimited xrp|flooding|emergency|delayed|moved to november', title + ' ' + body[:450]):
            return 'Negative_FUD', .8, 'The coverage centers on a technical vulnerability or emergency remediation; remediation does not imply an ongoing exploit.', flags
        return 'Neutral_Market_News', .78, 'The excerpt describes a routine technical release or security improvement, without a demonstrated incident.', flags

    if available and re.search(r'delay.*penalty|lower.*penalty|cites.*settlement', title):
        return 'Neutral_Market_News', .77, 'The excerpt reports a procedural penalty/settlement development, rather than a fresh adverse incident.', flags
    if re.search(r'fine.*(?:conclude|end).*lawsuit|lawsuit.*(?:conclude|end).*fine', title):
        return REVIEW, .45, 'A financial penalty and claimed legal closure are combined; the positive/negative legal interpretation needs review.', flags + ['mixed_legal_outcomes']
    incident = re.search(r'\bhack(?:ers?|ed)?\b|scam|\boutage\b|stole|stolen|vulnerability|unlimited xrp bug|moves to appeal|sec rejects|judge denies|judge.*denies|penalty|\bfine\b|remedies (?:fight|battle)', title)
    if incident:
        if re.search(r'debunk|no (?:hack|outage|scam)|false alarm|not (?:a |an )?(?:hack|outage)', title + ' ' + body[:200]):
            return REVIEW, .45, 'The incident language includes a denial or debunking; do not convert it into a confirmed negative event.', flags + ['negated_or_contested_incident']
        if re.search(r'after judge orders|as.*judge.*orders', title):
            return REVIEW, .45, 'The headline mixes a court penalty with market gains.', flags + ['mixed_legal_and_market']
        return 'Negative_FUD', evidence, 'The central subject is a reported security loss, operational disruption, penalty or adverse legal action; the claim remains unverified.', flags

    if re.search(r'predict|price target|moonshot|explod|astronomical|price.*headed|will.*outperform|headed to zero|could shatter|xrp to \$|price.*\$\d+.*by 20|price will touch|violent.*repricing|rally.*in the cards', title):
        if re.search(r'prediction|price prediction|price targets?', title) and re.search(r'elliott wave|support|resistance|breakout', body) and not re.search(r'moonshot|astronomical|guaranteed', title):
            return REVIEW, .48, 'A target is embedded in conditional technical analysis; speculation versus ordinary market analysis needs review.', flags + ['forecast_vs_technical_analysis']
        if re.search(r'predict.*(?:regulatory|sea change|more spot etfs|crypto adoption wave)', title):
            return REVIEW, .45, 'An industry/regulatory prediction is not an observed deployment or a clear token-price category.', flags + ['industry_outlook_uncertain']
        if re.search(r'will xrp hit|xrp could hit', title) and re.search(r'market cap|coin supply|circulating supply', body):
            return REVIEW, .48, 'A future target is evaluated using valuation constraints; the boundary between balanced analysis and speculation remains ambiguous.', flags + ['forecast_vs_valuation_analysis']
        if re.search(r'license.*price|banking.*price', title):
            return REVIEW, .45, 'The title mixes regulatory status and a speculative token-price benefit.', flags + ['mixed_regulatory_and_price']
        return 'Hype_Speculation', evidence, 'The headline centers on an attributed forecast, future target or extrapolated return rather than an observed outcome. This is a content label, not endorsement of the target.', flags

    neutral = re.search(r'\betf|inflow|outflow|open interest|support|resistance|liquidat|moving average|\brsi\b|\bmvrv\b|profit.to.loss|price (?:analysis|falls|faces|risks|recovery|struggles)|trading|market cap|whale|wallet growth|escrow|network activity|account|transaction|reclaims|below|slips|plunges|losers|declin|withdrawals|reserves|historical|price.*surges|price.*rally|\bup \d|\bdown \d|correlation|surges by|demand.*rebound', title)
    if neutral:
        if available or re.search(r'etf|inflows|outflows|open interest|escrow|price analysis|\brsi\b|market cap|\bup \d|\bdown \d', title):
            if 'rlusd hits $' in title and 'capitalization' in body:
                flags.append('headline_price_vs_market_cap_units')
            return 'Neutral_Market_News', evidence, 'The subject is market/on-chain measurements or conditional technical analysis. A fall in price alone is not classified as a security or legal incident.', flags
    if available and re.search(r'approval|authori[sz]ation|license|charter|regulat|files for|filing|court|legal|remedies|sec|denies|bank partner count|not mean.*banks|schwartz.*origins|quantum', title):
        return 'Neutral_Market_News', .78, 'The excerpt describes regulatory, corporate or explanatory developments without establishing a deployed use case or a new adverse incident.', flags
    if available and re.search(r'acquir|acquisition|investment|invests|donat|treasury|vote|stake|holding|shares|step down|sold|sales|sell.off|stablecoin.*growth', title):
        return 'Neutral_Market_News', .77, 'The excerpt concerns a corporate transaction, treasury/disclosure or observed business metric. It does not by itself establish XRPL deployment or token-price gains.', flags
    return REVIEW, .4, 'The available evidence does not support a single category under the content rubric.', flags + ['insufficient_or_mixed_evidence']


def classify_research(row, context, previous=None, saved=None):
    raw = context.get('excerpt', '') if context and matching(row, context) and context.get('status') == 'retrieved_unverified' else ''
    excerpt = clean_excerpt(row['headline'], raw)
    label, score, reason, flags = assess(row['headline'], excerpt)
    method = VERSION
    # Retain valid earlier AI assessments where the expanded rubric abstains. Never
    # let a stale proposal or an old AI status manufacture a human annotation.
    candidate = None
    if previous and matching(row, previous) and previous.get('predicted_label') in LABELS:
        candidate = (previous['predicted_label'], float(previous['heuristic_confidence']), previous['reasoning'])
    elif saved and matching(row, saved) and saved.get('suggested_label') in LABELS:
        candidate = (saved['suggested_label'], float(saved['confidence_estimate']), saved['reasoning'])
    if candidate and (not math.isfinite(candidate[1]) or not 0 <= candidate[1] <= 1):
        candidate = None
        flags.append('invalid_cached_confidence')
    if candidate and excerpt:
        if label == REVIEW and set(flags).issubset({'insufficient_or_mixed_evidence', 'source_text_encoding_artifact', 'sensational_framing_not_fact_checked'}):
            label, score, old_reason = candidate
            reason = 'Retained matching earlier provisional assessment: ' + old_reason
            method += '+cached_assessment'
        elif label in LABELS and label != candidate[0]:
            flags.append('changed_from_previous_ai_label')
            reason += f' Earlier AI suggestion was {candidate[0]}; disagreement requires human checking.'
            label, score = REVIEW, min(score, .49)
    if not excerpt:
        flags.append('headline_only_no_substantive_context')
        reason += ' Based on the headline only; the stored excerpt is missing or adds no substantive evidence.'
    else:
        reason += ' Evidence excerpt: ' + ' '.join(excerpt.split()[:28]) + '...'
    reason += ' No independent factual verification. Confidence is a heuristic score, not a probability.'
    return label, score, reason, flags, excerpt, method
