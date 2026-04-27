"""
Stage 3 — LDA topic modelling on high-confidence negative tweets.

Following Sia, Dalmia and Mielke (2020), we filter by sentiment first to
sharpen topic separation. Optimal K is selected by Röder, Both and
Hinneburg's (2015) c_v coherence over K in {5, 7, 10, 12}.

Input:  data/tweets_sentiment.parquet
Output: data/lda_topics.parquet
        data/lda_topics_top_words.csv
        data/lda_coherence.csv
        figures/lda_interactive.html (pyLDAvis)
"""
from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Optional dep — only needed for the interactive HTML
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    HAVE_PYLDAVIS = True
except Exception:
    HAVE_PYLDAVIS = False

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.80
K_CANDIDATES = [5, 7, 10, 12]
RANDOM_STATE = 42

EXTRA_STOPWORDS = {
    "tesco", "tescos", "u", "im", "ive", "th", "rt", "amp", "would",
    "could", "still", "get", "got", "go", "going", "gone", "one", "two",
    "see", "say", "said", "tell", "told", "really", "much", "even",
    "though", "always", "never", "today", "yesterday", "tomorrow",
    "back", "thing", "way", "make", "made", "ill", "thats", "dont",
    "doesnt", "didnt", "wasnt", "isnt", "wont", "youre", "theyre",
    "theyve", "theyd", "well", "people", "want", "need", "lol",
}


def ensure_nltk() -> None:
    for pkg in ("stopwords", "wordnet"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def make_tokeniser():
    sw = set(stopwords.words("english")) | EXTRA_STOPWORDS
    lemma = WordNetLemmatizer()
    word_re = re.compile(r"[a-z]{3,}")

    def tokenise(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#", " ", text)
        toks = word_re.findall(text)
        return [lemma.lemmatize(t) for t in toks if t not in sw and len(t) > 2]

    return tokenise


def main() -> None:
    ensure_nltk()
    df = pd.read_parquet(DATA_DIR / "tweets_sentiment.parquet")

    neg = df[(df["sentiment"] == "negative") & (df["sentiment_conf"] >= CONF_THRESHOLD)].copy()
    print(f"[load] {len(df):,} total -> {len(neg):,} high-confidence negatives")

    tokenise = make_tokeniser()
    neg["tokens"] = neg["clean_text"].map(tokenise)
    neg = neg[neg["tokens"].map(len) >= 4].copy()
    print(f"[clean] {len(neg):,} negatives after token filter (>=4 tokens)")

    docs = neg["tokens"].tolist()
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]
    print(f"[dict] vocab size = {len(dictionary)}")

    coherences = []
    best_k = None
    best_model = None
    best_score = -1.0
    for k in K_CANDIDATES:
        print(f"[lda] training k={k} ...")
        m = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=8,
            iterations=200,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=m, texts=docs, dictionary=dictionary, coherence="c_v"
        )
        score = cm.get_coherence()
        coherences.append({"k": k, "c_v": score})
        print(f"       c_v = {score:.4f}")
        if score > best_score:
            best_score, best_k, best_model = score, k, m

    pd.DataFrame(coherences).to_csv(DATA_DIR / "lda_coherence.csv", index=False)
    print(f"[lda] best k = {best_k} (c_v = {best_score:.4f})")

    # ------------------------------------------------------------------
    # Top words
    # ------------------------------------------------------------------
    rows = []
    for tid in range(best_k):
        top = best_model.show_topic(tid, topn=15)
        rows.append({
            "topic": tid,
            "top_words": ", ".join(w for w, _ in top),
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "lda_topics_top_words.csv", index=False)

    # ------------------------------------------------------------------
    # Doc-topic assignment
    # ------------------------------------------------------------------
    dom_topic, dom_prob = [], []
    for bow in corpus:
        dist = best_model.get_document_topics(bow, minimum_probability=0.0)
        t, p = max(dist, key=lambda x: x[1])
        dom_topic.append(t)
        dom_prob.append(p)
    neg["dominant_topic"] = dom_topic
    neg["topic_prob"] = dom_prob
    neg[[
        "id_str", "created_at", "clean_text", "sentiment_score",
        "dominant_topic", "topic_prob",
    ]].to_parquet(DATA_DIR / "lda_topics.parquet", index=False)

    # ------------------------------------------------------------------
    # pyLDAvis (Appendix B in report)
    # ------------------------------------------------------------------
    if HAVE_PYLDAVIS:
        try:
            vis = gensimvis.prepare(best_model, corpus, dictionary, sort_topics=False)
            pyLDAvis.save_html(vis, str(FIG_DIR / "lda_interactive.html"))
            print(f"[viz] wrote {FIG_DIR / 'lda_interactive.html'}")
        except Exception as e:
            print(f"[viz] pyLDAvis failed: {e}")
    else:
        print("[viz] pyLDAvis not installed — skipping interactive HTML")


if __name__ == "__main__":
    main()
