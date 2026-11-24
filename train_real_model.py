"""Train TF-IDF from human-labeled REAL headlines; chronological evaluation.

Command: python train_real_model.py
Saves an SEPARATE model so your synthetic baseline stays untouched.
No synthetic headlines are read or used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'xrp_real_labeled.csv'
ARTIFACTS = ROOT / 'artifacts'
DATA = ROOT / 'data'
LABELS = ['Real_Adoption', 'Hype_Speculation', 'Negative_FUD', 'Neutral_Market_News']


def human_labeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy manual labels remain valid; explicitly AI-origin rows never qualify.

    Suggestions have no `label` column and are never read by this trainer.
    Exported AI-assisted decisions require a human reviewer and timestamp.
    """
    eligible = df['label'].isin(LABELS)
    for column in ('label_status', 'label_verification_status', 'verification_status'):
        if column in df.columns:
            eligible &= ~df[column].fillna('').str.upper().eq('AI_PROVISIONAL')
    if 'label_origin' in df.columns:
        origin = df['label_origin'].fillna('')
        eligible &= origin.isin(['', 'human_manual', 'human_verified', 'human_reviewed_ai'])
        assisted = origin.eq('human_reviewed_ai')
        for column in ('human_reviewed_by', 'human_reviewed_at'):
            evidence = df[column].fillna('').str.strip().ne('') if column in df else False
            eligible &= ~assisted | evidence
    return df[eligible].copy()


def load_labeled() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError('No real labeled dataset. Run collect_real_news.py then label_real_news.py')
    df = pd.read_csv(SOURCE, dtype=str).fillna('')
    needed = {'date', 'headline', 'source', 'label', 'id'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing columns: {needed - set(df.columns)}')
    df = human_labeled_rows(df)
    df = df[~df['source'].str.contains('SYNTHETIC', case=False, na=False)]
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date']).copy()
    df['norm'] = (df['headline'].str.lower()
                  .str.replace(r'[^a-z0-9 ]', ' ', regex=True)
                  .str.replace(r'\s+', ' ', regex=True).str.strip())
    df = df[df['norm'].str.len() >= 12]
    df = df.sort_values('date').drop_duplicates('norm', keep='first').drop_duplicates('id')
    df = df.sort_values('date').reset_index(drop=True)
    counts = df['label'].value_counts()
    if len(df) < 120 or any(counts.get(label, 0) < 20 for label in LABELS):
        raise ValueError(f'Need >=120 distinct, human-labeled articles and >=20 for EACH of 4 classes. Now: {len(df)}; {counts.to_dict()}. Collect and label more REAL news first.')
    return df


def split_chronologically(df: pd.DataFrame):
    """70/15/15 by time, with a 24h embargo between partitions."""
    n = len(df)
    cut1, cut2 = int(n * .70), int(n * .85)
    boundary1 = df.iloc[cut1]['date']
    boundary2 = df.iloc[cut2]['date']
    train = df.iloc[:cut1].copy()
    val = df.iloc[cut1:cut2].copy()
    test = df.iloc[cut2:].copy()
    train = train[train['date'] < boundary1 - pd.Timedelta(hours=24)]
    val = val[(val['date'] >= boundary1) &
              (val['date'] < boundary2 - pd.Timedelta(hours=24))]
    test = test[test['date'] >= boundary2]
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        counts = frame['label'].value_counts()
        missing = set(LABELS) - set(counts.index)
        if len(frame) < 10 or missing:
            raise ValueError(f'{name} lacks enough chronologically separate, diverse data: {counts.to_dict()}. Label more dates/articles.')
    return train, val, test


def fit_and_evaluate() -> dict:
    # Prepare deduplicated, embargoed partitions; refuse insufficient data BEFORE fitting.
    # The optional provisional file is never loaded by this standard evaluation.
    from ai_labeling import workflow_lock
    from prepare_real_training import load_verified_partitions, prepare
    with workflow_lock(DATA):
        prepare(DATA)
        train, val, test = load_verified_partitions(DATA, DATA / 'training_prepared')
    df = pd.concat([train, val, test], ignore_index=True)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=12000,
                                  sublinear_tf=True, min_df=1, strip_accents='unicode')),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=2500,
                                           random_state=42)),
    ])
    model.fit(train['headline'], train['label'])
    # Validation is evaluated without fitting; future hyperparameters would be chosen here.
    validation_pred = model.predict(val['headline'])
    predicted = model.predict(test['headline'])
    proba = model.predict_proba(test['headline']).max(axis=1)
    cm = confusion_matrix(test['label'], predicted, labels=LABELS)
    report = classification_report(test['label'], predicted, labels=LABELS,
                                   zero_division=0, output_dict=True)
    metrics = {
        'dataset': 'real headlines, manually labeled, GDELT first-seen timestamps',
        'warning': 'Not an estimate of 24-hour price direction accuracy.',
        'counts': {'all': len(df), 'train': len(train), 'validation': len(val), 'test': len(test)},
        'periods_utc': {'train_end': train['date'].max().isoformat(),
                        'validation_end': val['date'].max().isoformat(),
                        'test_start': test['date'].min().isoformat()},
        'accuracy': float(accuracy_score(test['label'], predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(test['label'], predicted)),
        'macro_f1': float(f1_score(test['label'], predicted, average='macro')),
        'validation_macro_f1': float(f1_score(val['label'], validation_pred, average='macro')),
        'confusion_matrix': cm.tolist(),
        'report': report,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / 'news_tfidf_real.joblib')
    (ARTIFACTS / 'metrics_tfidf_real.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(ax=ax, cmap='Blues')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / 'confusion_matrix_tfidf_real.png', dpi=140)
    plt.close(fig)
    results = test[['id', 'date', 'headline', 'source']].copy()
    results['true_label'] = test['label'].to_numpy()
    results['predicted_label'] = predicted
    results['confidence_pct'] = np.round(100 * proba, 2)
    results['held_out'] = True
    results.to_csv(DATA / 'real_news_test_predictions.csv', index=False)
    print(f'\nREAL headline test: n={len(test)}; accuracy={metrics["accuracy"]:.1%}, '
          f'balanced accuracy={metrics["balanced_accuracy"]:.1%}, macro-F1={metrics["macro_f1"]:.1%}')
    print(classification_report(test['label'], predicted, labels=LABELS, zero_division=0))
    print('Saved model, report, confusion matrix and ONLY holdout predictions.')
    return metrics


if __name__ == '__main__':
    fit_and_evaluate()
