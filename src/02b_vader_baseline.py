"""
Stage 2b — VADER baseline for sentiment, kept on the same 92.7k tweets so
the report can compare the two models.

VADER (Hutto and Gilbert, 2014) is a rule/lexicon hybrid; it costs nothing
to run and is the standard baseline in the literature. We compare its
labels against RoBERTa using Cohen's kappa and a 3x3 agreement matrix.

Output:
    data/tweets_vader.parquet
    data/model_agreement.csv
    data/model_agreement_summary.txt
"""
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DATA = Path(__file__).resolve().parent.parent / "data"

# VADER thresholds from Hutto & Gilbert (2014)
POS_T = 0.05
NEG_T = -0.05


def vader_label(compound: float) -> str:
    if compound >= POS_T:
        return "positive"
    if compound <= NEG_T:
        return "negative"
    return "neutral"


def main() -> None:
    df = pd.read_parquet(DATA / "tweets_sentiment.parquet")
    sia = SentimentIntensityAnalyzer()

    # vectorise via simple list-comp; VADER on 93k texts takes < 1 minute
    print(f"running VADER over {len(df):,} tweets...")
    scores = [sia.polarity_scores(t)["compound"] for t in df["clean_text"].tolist()]
    df["vader_compound"] = scores
    df["vader_label"] = [vader_label(s) for s in scores]

    df.to_parquet(DATA / "tweets_vader.parquet", index=False)

    # agreement
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(df["sentiment"], df["vader_label"], labels=labels)
    kappa = cohen_kappa_score(df["sentiment"], df["vader_label"])
    pct_agree = (df["sentiment"] == df["vader_label"]).mean()

    cm_df = pd.DataFrame(cm, index=[f"roberta_{l}" for l in labels],
                          columns=[f"vader_{l}" for l in labels])
    cm_df.to_csv(DATA / "model_agreement.csv")

    summary = (
        f"Model agreement (RoBERTa vs VADER), n = {len(df):,}\n"
        f"=================================================\n"
        f"% agreement   : {pct_agree*100:.2f}%\n"
        f"Cohen's kappa : {kappa:.4f}\n\n"
        f"Confusion matrix (rows = RoBERTa, cols = VADER):\n"
        f"{cm_df.to_string()}\n\n"
        f"VADER class shares:\n"
        f"{df['vader_label'].value_counts(normalize=True).round(4).to_string()}\n"
    )
    (DATA / "model_agreement_summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
