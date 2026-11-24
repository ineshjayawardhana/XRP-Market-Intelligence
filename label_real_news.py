"""Human labeling of REAL XRP headlines, persists decisions after every answer.

Command: python label_real_news.py --count 50
Keys: 1=adoption 2=hype 3=negative 4=neutral s=skip q=quit.
Labels describe headline CONTENT, not factual verification or market direction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'xrp_real_news_unlabeled.csv'
OUTPUT = ROOT / 'data' / 'xrp_real_labeled.csv'
LABELS = {'1': 'Real_Adoption', '2': 'Hype_Speculation',
          '3': 'Negative_FUD', '4': 'Neutral_Market_News'}


def annotate(count: int = 50, review_skipped: bool = False) -> None:
    if not SOURCE.exists():
        raise FileNotFoundError('First run: python collect_real_news.py --days 60')
    all_df = pd.read_csv(SOURCE, dtype=str).fillna('')
    if OUTPUT.exists():
        reviewed = pd.read_csv(OUTPUT, dtype=str).fillna('')
        annotated = dict(zip(reviewed['id'], reviewed['label']))
        all_df['label'] = all_df['id'].map(annotated).fillna(all_df['label'])
    all_df['label'] = all_df['label'].fillna('')
    pending = all_df.index[all_df['label'].eq('SKIP' if review_skipped else '')].tolist()
    if not pending:
        if review_skipped:
            print('No SKIP headlines to review. Existing labels are unchanged.')
        else:
            print('No new unlabeled headlines. Use --review-skipped to reconsider SKIP rows.')
        return
    print('\nLABEL GUIDE: 1 = verified-looking XRPL product/deployment/integration content;')
    print('2 = sensational price targets, rumors, speculation, misleading certainty;')
    print('3 = outages, legal/regulatory risks, negative news or safety incidents.')
    print('4 = ordinary/neutral market reporting, balanced price analysis, factual ETF flows.')
    print('s = skip unclear, unrelated, or mixed headline (never force a label).')
    if review_skipped:
        print('REVIEW MODE: showing previously SKIP headlines only. Original non-SKIP labels are preserved.')
    print('Do not inspect future price changes when deciding labels.\n')
    processed = 0
    for idx in pending:
        row = all_df.loc[idx]
        print(f'\n[{processed+1}/{min(len(pending), count)}] {row["date"]} — {row["source"]}')
        print('HEADLINE:', row['headline'])
        print('ARTICLE:', row['url'])
        answer = input('1=Adoption 2=Hype 3=Negative 4=Neutral s=Skip q=Quit > ').strip().lower()
        if answer == 'q':
            break
        if answer not in (*LABELS.keys(), 's'):
            print('Invalid input. This headline is left unchanged.')
            continue
        all_df.at[idx, 'label'] = LABELS.get(answer, 'SKIP')
        # Save after EVERY answer to avoid losing hours of human annotation.
        temporary = OUTPUT.with_suffix('.tmp')
        temporary.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(temporary, index=False, encoding='utf-8-sig')
        temporary.replace(OUTPUT)
        processed += 1
        if processed >= count:
            break
    counts = all_df['label'].value_counts().to_dict()
    print('Saved:', OUTPUT, '| labels:', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--review-skipped', action='store_true',
                        help='Review earlier SKIP labels without changing other reviewed rows')
    args = parser.parse_args()
    annotate(args.count, review_skipped=args.review_skipped)
