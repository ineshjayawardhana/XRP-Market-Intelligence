"""Phase 1: create synthetic headlines, train and evaluate a news classifier.

All sample rows are SYNTHETIC, not verified published XRP articles.
Usage: python model.py --train
       python model.py --train-distilbert    (optional, after extra dependencies)
"""
from __future__ import annotations

import argparse
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "xrp_headlines_200.csv"
ARTIFACTS = ROOT / "artifacts"
LABELS = ["Real_Adoption", "Hype_Speculation", "Negative_FUD"]

# 40 + 35 + 25 = 100 fictional training headline ideas; two paraphrases each.
REAL = [
    "XRPL developer releases production-ready payment gateway integration",
    "Regional remittance provider pilots XRP Ledger settlement service",
    "Enterprise software team integrates XRPL payment confirmations",
    "New wallet provider adds XRP Ledger transaction support",
    "Payment startup tests cross-border transfers on XRPL",
    "Open-source team publishes an audited XRPL integration library",
    "Fintech firm connects customer payout system to XRP Ledger",
    "Blockchain infrastructure provider expands XRPL node capacity",
    "Merchant service launches XRP Ledger invoice settlement option",
    "International payment processor completes XRPL technical pilot",
    "University research team deploys an XRPL verification prototype",
    "Digital asset custodian announces support for XRPL accounts",
    "Payments company adds XRP Ledger to supported networks",
    "Institutional custody platform integrates RLUSD token support",
    "Banking technology vendor tests token transfers on XRPL",
    "Regulated service provider launches an XRPL wallet feature",
    "Remittance application expands XRP Ledger payout functionality",
    "Developer tools project releases XRPL monitoring dashboard",
    "Retail payment processor deploys an XRPL checkout plugin",
    "Corporate treasury software supports XRPL transaction tracking",
    "Financial infrastructure startup launches XRPL test integration",
    "Tokenization project deploys assets using XRP Ledger technology",
    "Audited smart payment adapter supports XRPL transactions",
    "API platform adds validated XRP Ledger transaction indexing",
    "Ledger infrastructure provider launches XRPL archival nodes",
    "Payment network connects a new XRPL settlement corridor",
    "Business wallet platform supports multisignature XRPL accounts",
    "Fintech integrator publishes a working XRPL settlement connector",
    "Cross-border payroll platform pilots XRPL transfers",
    "Digital identity prototype writes verification records to XRPL",
    "Payment aggregator integrates XRPL transaction status alerts",
    "Merchant accounting software adds XRP Ledger reconciliation",
    "Custody software vendor supports XRPL token balances",
    "Logistics platform pilots XRP Ledger payment verification",
    "Blockchain analytics provider expands XRPL data coverage",
    "Payment operations team tests RLUSD settlement workflows",
    "Institutional wallet vendor adds secure XRPL signing feature",
    "Open-source contributor releases XRPL developer SDK update",
    "Billing service integrates XRP Ledger payment references",
    "Cross-border fintech consortium completes XRPL pilot program",
]
HYPE = [
    "XRP will reach 1000 dollars tomorrow according to viral prediction",
    "Social media claims XRP is guaranteed to skyrocket tonight",
    "Influencer says XRP will make every holder a millionaire",
    "Viral post predicts XRP price explosion without evidence",
    "XRP to 625 dollars soon says anonymous online account",
    "Traders online call XRP the next guaranteed 100x coin",
    "Unverified rumor says XRP will moon before midnight",
    "Influencer predicts unstoppable XRP rally this weekend",
    "Speculative post claims an XRP mega pump is inevitable",
    "Online personality announces secret XRP price target",
    "Rumor claims a celebrity will send XRP prices to the moon",
    "Anonymous chart screenshot predicts XRP price multiplication",
    "Viral message tells followers that XRP gains are guaranteed",
    "Meme account posts XRP rocket countdown and price target",
    "XRP holders urged to get rich quick by social media thread",
    "Unconfirmed whisper predicts XRP becoming world's top asset",
    "Viral claim says XRP cannot fall after next announcement",
    "Influencer expects triple-digit XRP price overnight",
    "Unverified XRP supply shock rumor sparks price speculation",
    "Meme trend says XRP will outperform every asset tomorrow",
    "Speculative video promises life-changing XRP profit within days",
    "Online forum claims XRP rally will never reverse",
    "Anonymous account calls XRP a risk-free investment",
    "Viral XRP moon prediction spreads without named sources",
    "Social media hype urges everyone to buy XRP immediately",
    "Rumor of a secret XRP partnership fuels wild target claims",
    "Influencer says XRP is certain to jump 500 percent tomorrow",
    "Unverified XRP ETF rumor sparks sensational price forecasts",
    "Online prediction expects XRP to reach 100 dollars instantly",
    "Meme community starts XRP to the moon countdown",
    "Clickbait headline forecasts a guaranteed XRP supercycle",
    "Speculative post calls XRP the easiest way to get rich",
    "Anonymous analyst predicts XRP 50x move in one day",
    "Viral social post claims XRP has zero downside risk",
    "FOMO message urges buying XRP before a rumored surprise",
]
NEGATIVE = [
    "XRPL infrastructure outage delays transaction processing",
    "Security researchers discover critical flaw in XRP wallet software",
    "Regulatory investigation creates uncertainty for XRP-related firm",
    "Exchange suspends XRP withdrawals during technical incident",
    "Court filing adds legal uncertainty to XRP market outlook",
    "Phishing campaign targets holders of XRP Ledger assets",
    "Network congestion slows XRP Ledger transaction confirmations",
    "Major wallet bug leads provider to pause XRP transfers",
    "Regulator raises compliance concerns over digital asset service",
    "False XRP partnership statement triggers public correction",
    "Exchange delists XRP trading pair following compliance review",
    "Audit identifies security issues in third-party XRPL software",
    "XRP market declines after negative regulatory headlines",
    "Service interruption affects XRP payment gateway users",
    "Institutional pilot using XRPL is canceled after review",
    "Fake XRP giveaway scam prompts investor safety warning",
    "Liquidity provider reports disruption in XRP market making",
    "Legal dispute creates additional uncertainty for XRP services",
    "Trading platform reports prolonged outage on XRP markets",
    "Exploit disclosure raises concerns for XRPL ecosystem tools",
    "Compliance rules force payment company to suspend XRP product",
    "XRPL integration rollout delayed by engineering problems",
    "Financial watchdog warns about misleading XRP promotions",
    "Suspicious token activity triggers XRPL service investigation",
    "Technical defect temporarily disables XRP wallet transactions",
]


def generate_dataset(force: bool = False) -> pd.DataFrame:
    """Make exactly 200 examples in four columns, and label the source as synthetic."""
    if DATA.exists() and not force:
        return pd.read_csv(DATA)
    assert (len(REAL), len(HYPE), len(NEGATIVE)) == (40, 35, 25)
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for label, topics in zip(LABELS, [REAL, HYPE, NEGATIVE]):
        for topic in topics:
            for phrase in [topic, f"Report discusses: {topic[0].lower() + topic[1:]}"]:
                rows.append({
                    "date": (start + timedelta(days=len(rows))).isoformat(),
                    "headline": phrase,
                    "source": "SYNTHETIC_TRAINING_EXAMPLE",
                    "label": label,
                })
    assert len(rows) == 200
    DATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA, index=False)
    return pd.DataFrame(rows)


def split_data(df: pd.DataFrame):
    """Split by TOPIC rather than near-duplicate row to reduce test leakage."""
    if len(df) == 200 and df["source"].eq("SYNTHETIC_TRAINING_EXAMPLE").all():
        group_ids = np.arange(len(df)) // 2
        groups = pd.DataFrame({"group": group_ids, "label": df["label"]}).drop_duplicates("group")
        train_g, test_g = train_test_split(
            groups, test_size=0.25, random_state=42, stratify=groups["label"]
        )
        train_df = df[np.isin(group_ids, train_g["group"])].copy()
        test_df = df[np.isin(group_ids, test_g["group"])].copy()
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df["label"]
        )
    return train_df, test_df


def save_metrics(y_true, y_pred, name: str) -> dict:
    from matplotlib import pyplot as plt

    ARTIFACTS.mkdir(exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                        zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "test_examples": len(y_true),
        "warning": "Synthetic-data evaluation; not verified real-world accuracy.",
    }
    (ARTIFACTS / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS,
                                            xticks_rotation=20, ax=ax, cmap="Blues")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"confusion_matrix_{name}.png", dpi=140)
    plt.close(fig)
    print(f"\n{name} accuracy on synthetic held-out headlines: {metrics['accuracy']:.2%}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true labels, columns=predicted labels):")
    print(np.asarray(metrics["confusion_matrix"]))
    return metrics


def train_model() -> dict:
    df = generate_dataset()
    required = {"date", "headline", "source", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must have columns: {', '.join(sorted(required))}")
    train_df, test_df = split_data(df)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                  stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=42)),
    ])
    model.fit(train_df["headline"].astype(str), train_df["label"])
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "news_tfidf.joblib")
    metrics = save_metrics(test_df["label"], model.predict(test_df["headline"]), "tfidf")
    print(f"Saved {len(df)} sample headlines to {DATA}")
    print("Note: model confidence is not a calibrated probability of truth.")
    return metrics


@lru_cache(maxsize=2)
def _load_tfidf():
    # Keep the original synthetic prototype intact. Switch explicitly after
    # REAL, manually-labeled, chronologically held-out evaluation succeeds.
    variant = os.getenv("NEWS_TFIDF_MODEL", "synthetic").strip().lower()
    if variant not in ("synthetic", "real"):
        raise ValueError("NEWS_TFIDF_MODEL must be 'synthetic' or 'real'")
    filename = "news_tfidf_real.joblib" if variant == "real" else "news_tfidf.joblib"
    path = ARTIFACTS / filename
    if not path.exists():
        command = "python train_real_model.py" if variant == "real" else "python model.py --train"
        raise FileNotFoundError(f"Missing {filename}. Run: {command}")
    return joblib.load(path)  # Only load model files YOU created or trust.


@lru_cache(maxsize=1)
def _load_distilbert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = ARTIFACTS / "distilbert_news"
    if not path.exists():
        raise FileNotFoundError("Train first: python model.py --train-distilbert")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model, torch


def predict_news(headline: str, backend: str = "tfidf") -> dict:
    """Returns {label, confidence}; confidence is classifier certainty, not market odds."""
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("Please enter a nonempty headline.")
    if backend == "distilbert":
        tokenizer, model, torch = _load_distilbert()
        encoded = tokenizer(headline, return_tensors="pt", truncation=True,
                            padding=True, max_length=96)
        with torch.no_grad():
            p = torch.softmax(model(**encoded).logits, dim=-1)[0].numpy()
        predicted_idx = int(p.argmax())
        return {"label": model.config.id2label[predicted_idx],
                "confidence": round(float(p[predicted_idx] * 100), 1)}
    if backend != "tfidf":
        raise ValueError("backend must be 'tfidf' or 'distilbert'")
    model = _load_tfidf()
    p = model.predict_proba([headline])[0]
    idx = int(p.argmax())
    return {"label": str(model.classes_[idx]),
            "confidence": round(float(p[idx] * 100), 1)}


def train_distilbert() -> dict:
    """Optional upgrade. Installs from requirements-bert.txt before running."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    df = generate_dataset()
    train_df, test_df = split_data(df)
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    class NewsDataset(Dataset):
        def __init__(self, frame):
            self.texts = frame["headline"].astype(str).tolist()
            self.labels = [LABELS.index(label) for label in frame["label"]]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True,
                                 max_length=96, return_tensors="pt")
            return {"input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id={name: i for i, name in enumerate(LABELS)},
    )
    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "distilbert_checkpoints"),
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=42,
        fp16=False,  # Works on CPU; no CUDA requirement.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=NewsDataset(train_df),
                      eval_dataset=NewsDataset(test_df))
    trainer.train()
    predictions = trainer.predict(NewsDataset(test_df)).predictions.argmax(axis=1)
    predicted = [LABELS[int(i)] for i in predictions]
    metrics = save_metrics(test_df["label"].tolist(), predicted, "distilbert")
    save_path = ARTIFACTS / "distilbert_news"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("DistilBERT saved to", save_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train TF-IDF baseline")
    parser.add_argument("--train-distilbert", action="store_true", help="Train transformer")
    parser.add_argument("--predict", type=str, help="Classify a single headline")
    args = parser.parse_args()
    if args.train_distilbert:
        train_distilbert()
    elif args.predict:
        print(predict_news(args.predict))
    else:
        train_model()
