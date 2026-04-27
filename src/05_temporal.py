"""
Stage 5 — Event-annotated temporal sentiment analysis.

For 2020 we anchor the sentiment timeline against verified UK retail and
COVID-19 events (sources cited in the report references). For each event
we compute a 14-day pre/post comparison of mean sentiment_score and the
share of negative tweets.

Note: this analysis is associational, not causal.

Input:  data/tweets_sentiment.parquet
Output: data/daily_sentiment.csv
        data/monthly_sentiment.csv
        data/event_study.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

WINDOW_DAYS = 14

EVENTS = [
    # (label, ISO date) — references in report bibliography
    ("WHO declares COVID-19 a pandemic", "2020-03-11"),
    ("UK national lockdown begins", "2020-03-23"),
    ("Tesco hires 35k temp staff", "2020-03-25"),
    ("Tesco opens priority slots for vulnerable", "2020-04-03"),
    ("First lockdown easing in England", "2020-06-15"),
    ("Tesco sells Tesco Lotus Asia ops", "2020-03-09"),
    ("Eat Out To Help Out begins", "2020-08-03"),
    ("Tier system announced", "2020-10-12"),
    ("Second national lockdown begins", "2020-11-05"),
    ("Tier 4 announced (London/SE)", "2020-12-19"),
]


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "tweets_sentiment.parquet")
    df["date"] = df["created_at"].dt.tz_convert("UTC").dt.date
    df["month"] = df["created_at"].dt.tz_convert("UTC").dt.to_period("M").astype(str)

    # daily
    daily = (
        df.groupby("date")
        .agg(
            n=("id_str", "count"),
            mean_score=("sentiment_score", "mean"),
            share_neg=("sentiment", lambda s: (s == "negative").mean()),
            share_pos=("sentiment", lambda s: (s == "positive").mean()),
        )
        .reset_index()
    )
    daily.to_csv(DATA_DIR / "daily_sentiment.csv", index=False)
    print(f"[save] {len(daily)} daily rows -> data/daily_sentiment.csv")

    # monthly
    monthly = (
        df.groupby("month")
        .agg(
            n=("id_str", "count"),
            mean_score=("sentiment_score", "mean"),
            share_neg=("sentiment", lambda s: (s == "negative").mean()),
            share_pos=("sentiment", lambda s: (s == "positive").mean()),
        )
        .reset_index()
    )
    monthly.to_csv(DATA_DIR / "monthly_sentiment.csv", index=False)
    print(f"[save] {len(monthly)} monthly rows -> data/monthly_sentiment.csv")

    # event study
    df["dt"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    rows = []
    for label, dt_str in EVENTS:
        evt = pd.Timestamp(dt_str)
        pre = df[(df["dt"] >= evt - pd.Timedelta(days=WINDOW_DAYS)) & (df["dt"] < evt)]
        post = df[(df["dt"] >= evt) & (df["dt"] < evt + pd.Timedelta(days=WINDOW_DAYS))]
        rows.append({
            "event": label,
            "date": dt_str,
            "n_pre": len(pre),
            "n_post": len(post),
            "mean_pre": round(pre["sentiment_score"].mean(), 4) if len(pre) else None,
            "mean_post": round(post["sentiment_score"].mean(), 4) if len(post) else None,
            "delta": (
                round(post["sentiment_score"].mean() - pre["sentiment_score"].mean(), 4)
                if len(pre) and len(post)
                else None
            ),
            "share_neg_pre": round((pre["sentiment"] == "negative").mean(), 4) if len(pre) else None,
            "share_neg_post": round((post["sentiment"] == "negative").mean(), 4) if len(post) else None,
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(DATA_DIR / "event_study.csv", index=False)
    print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
