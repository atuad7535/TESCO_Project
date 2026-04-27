"""
Stage 1 — Load and clean the Tesco Twitter corpus.

Input:  tesco.json (pandas to_json(orient='columns'); ~489 MB; 96,702 rows)
Output: data/tweets_clean.parquet
        data/cleaning_report.txt

Steps applied:
  1. Read JSON with json.load (the file is a single column-oriented object)
  2. Cast created_at (epoch ms) -> UTC pandas Timestamp
  3. Filter to English-language tweets (lang == 'en')
  4. Drop empty / duplicate id_str rows
  5. Reconstruct full text from extended_tweet.full_text where text is truncated
  6. Flatten user.screen_name and entities.user_mentions to top-level columns
  7. Add boolean flags is_retweet / is_reply / is_quote for downstream stages
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def load_columnar_json(path: Path) -> pd.DataFrame:
    """Load a pandas-style column-oriented JSON file into a DataFrame."""
    print(f"[load] reading {path} ...")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # Build DataFrame; index is the row-id key set
    df = pd.DataFrame(raw)
    print(f"[load] raw rows = {len(df):,}, columns = {len(df.columns)}")
    return df


def reconstruct_full_text(row: pd.Series) -> str:
    """Use extended_tweet.full_text if the visible text is truncated."""
    ext = row.get("extended_tweet")
    if isinstance(ext, dict):
        ft = ext.get("full_text")
        if isinstance(ft, str) and ft.strip():
            return ft
    return row.get("text") or ""


def normalise_text(text: str) -> str:
    """Strip URLs and collapse whitespace; keep handles + hashtags for SNA/NLP."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def extract_user_mentions(entities) -> list[str]:
    if not isinstance(entities, dict):
        return []
    mentions = entities.get("user_mentions") or []
    return [m.get("screen_name") for m in mentions if isinstance(m, dict) and m.get("screen_name")]


def extract_hashtags(entities) -> list[str]:
    if not isinstance(entities, dict):
        return []
    tags = entities.get("hashtags") or []
    return [t.get("text", "").lower() for t in tags if isinstance(t, dict) and t.get("text")]


def main(input_path: Path) -> None:
    df = load_columnar_json(input_path)

    # ------------------------------------------------------------------
    # Timestamp + window
    # ------------------------------------------------------------------
    df["created_at"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
    n_initial = len(df)
    print(f"[clean] date range: {df['created_at'].min()} -> {df['created_at'].max()}")

    # ------------------------------------------------------------------
    # Language filter
    # ------------------------------------------------------------------
    df = df[df["lang"] == "en"].copy()
    n_after_lang = len(df)
    print(f"[clean] dropped {n_initial - n_after_lang:,} non-English tweets")

    # ------------------------------------------------------------------
    # Reconstruct full text and normalise
    # ------------------------------------------------------------------
    df["full_text"] = df.apply(reconstruct_full_text, axis=1)
    df["clean_text"] = df["full_text"].map(normalise_text)
    df = df[df["clean_text"].str.len() >= 5].copy()
    n_after_text = len(df)
    print(f"[clean] dropped {n_after_lang - n_after_text:,} empty/short-text rows")

    # ------------------------------------------------------------------
    # Deduplicate by tweet id_str
    # ------------------------------------------------------------------
    df = df.drop_duplicates(subset=["id_str"], keep="first")
    n_after_dedup = len(df)
    print(f"[clean] dropped {n_after_text - n_after_dedup:,} duplicate ids")

    # ------------------------------------------------------------------
    # Flatten user / entities
    # ------------------------------------------------------------------
    df["author_screen_name"] = df["user"].map(
        lambda u: u.get("screen_name") if isinstance(u, dict) else None
    )
    df["author_followers"] = df["user"].map(
        lambda u: u.get("followers_count") if isinstance(u, dict) else None
    )
    df["mentions"] = df["entities"].map(extract_user_mentions)
    df["hashtags"] = df["entities"].map(extract_hashtags)

    df["is_retweet"] = df["retweeted_status"].notna()
    df["is_quote"] = df["quoted_status"].notna()
    df["is_reply"] = df["in_reply_to_status_id"].notna()

    # Numeric coercions
    for col in ("retweet_count", "favorite_count", "reply_count", "quote_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Keep only the columns we actually need downstream — keeps parquet small
    # ------------------------------------------------------------------
    keep_cols = [
        "id_str", "created_at", "clean_text", "full_text",
        "author_screen_name", "author_followers",
        "mentions", "hashtags",
        "is_retweet", "is_quote", "is_reply",
        "in_reply_to_screen_name", "in_reply_to_status_id_str",
        "retweet_count", "favorite_count", "reply_count", "quote_count",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].reset_index(drop=True)

    out_path = DATA_DIR / "tweets_clean.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[save] wrote {len(out):,} rows -> {out_path}")

    # ------------------------------------------------------------------
    # Cleaning audit log
    # ------------------------------------------------------------------
    report = (
        f"Cleaning report\n"
        f"================\n"
        f"raw rows           : {n_initial:,}\n"
        f"after lang=='en'   : {n_after_lang:,}\n"
        f"after text>=5 chars: {n_after_text:,}\n"
        f"after dedup id_str : {n_after_dedup:,}\n"
        f"final rows         : {len(out):,}\n"
        f"\n"
        f"Type breakdown (final):\n"
        f"  original tweets : {(~out['is_retweet']).sum():,}\n"
        f"  retweets        : {out['is_retweet'].sum():,}\n"
        f"  replies         : {out['is_reply'].sum():,}\n"
        f"  quotes          : {out['is_quote'].sum():,}\n"
        f"\n"
        f"Date range: {out['created_at'].min()} -> {out['created_at'].max()}\n"
    )
    (DATA_DIR / "cleaning_report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Downloads" / "tesco.json",
        help="Path to tesco.json",
    )
    args = parser.parse_args()
    main(args.input)
