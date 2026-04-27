"""
Stage 2 — Sentiment classification.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest (Loureiro et al., 2022),
chosen because it was fine-tuned on ~124M tweets and outputs three classes
(negative / neutral / positive) — directly aligned with our downstream LDA
filter and event-study analysis.

Apple-Silicon (MPS) acceleration is used when available; otherwise CPU.

Input:  data/tweets_clean.parquet
Output: data/tweets_sentiment.parquet
        data/sentiment_summary.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]
SCORE_MAP = {"negative": -1, "neutral": 0, "positive": 1}
BATCH_SIZE = 64
MAX_TOKENS = 128


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "tweets_clean.parquet")
    print(f"[load] {len(df):,} tweets to classify")

    device = get_device()
    print(f"[model] loading {MODEL_NAME} on {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    texts = df["clean_text"].tolist()
    n = len(texts)
    pred_label = np.empty(n, dtype=object)
    pred_conf = np.empty(n, dtype=np.float32)
    pred_score = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, BATCH_SIZE), desc="sentiment"):
            chunk = texts[start : start + BATCH_SIZE]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            idx = probs.argmax(axis=1)
            for i, ix in enumerate(idx):
                lbl = LABELS[ix]
                pred_label[start + i] = lbl
                pred_conf[start + i] = float(probs[i, ix])
                # signed score = P(pos) - P(neg) -> [-1, 1]
                pred_score[start + i] = float(probs[i, 2] - probs[i, 0])

    df["sentiment"] = pred_label
    df["sentiment_conf"] = pred_conf
    df["sentiment_score"] = pred_score

    out = DATA_DIR / "tweets_sentiment.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    counts = df["sentiment"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    summary = (
        f"Sentiment summary (n = {len(df):,})\n"
        f"================================\n"
        f"negative : {counts.get('negative', 0):>7,}  ({pct.get('negative', 0)}%)\n"
        f"neutral  : {counts.get('neutral', 0):>7,}  ({pct.get('neutral', 0)}%)\n"
        f"positive : {counts.get('positive', 0):>7,}  ({pct.get('positive', 0)}%)\n"
        f"\n"
        f"mean sentiment_score = {df['sentiment_score'].mean():.4f}\n"
        f"mean confidence      = {df['sentiment_conf'].mean():.4f}\n"
    )
    (DATA_DIR / "sentiment_summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
