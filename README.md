# EFIMM0139 — Tesco Twitter Analytics (2020 COVID Year)

Coursework for *Social Media and Web Analytics*, University of Bristol.
Three-technique pipeline applied to **96,702 Tesco-related tweets from January–December 2020**:
sentiment classification (RoBERTa), latent topic discovery (LDA on negative-sentiment
sub-corpus) and social-network analysis (mention/reply/retweet graph).

## Repository layout

```
efimm0139_tesco/
├── data/                # Parquet/CSV intermediates (gitignored)
├── figures/             # All PNG figures referenced in the report
├── src/
│   ├── 01_load_clean.py     # JSON → cleaned DataFrame
│   ├── 02_sentiment.py      # RoBERTa / VADER sentiment classification
│   ├── 03_topic_lda.py      # LDA on negative tweets, c_v coherence, pyLDAvis
│   ├── 04_sna.py            # NetworkX graph, centrality, Louvain communities
│   ├── 05_temporal.py       # Event-annotated monthly sentiment timeline
│   └── 06_make_figures.py   # Reproduce all report figures from saved data
├── report.md            # 3,000-word report (Harvard referencing)
├── report.docx          # Submission-ready Word version (generated via pandoc)
└── requirements.txt
```

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
# place tesco.json in ../tesco.json (relative to src/) or pass --input
python src/01_load_clean.py --input ~/Downloads/tesco.json
python src/02_sentiment.py
python src/03_topic_lda.py
python src/04_sna.py
python src/05_temporal.py
python src/06_make_figures.py
pandoc report.md -o report.docx
```

## Data ethics & provenance

The dataset is a publicly-shared academic corpus of UK supermarket-related tweets
(`tesco.json`, 489 MB, ~96.7 k records, January–December 2020) made available via
the unit-shared Google Drive. Per the unit handbook the dataset qualifies as an
*established dataset*; its source, scope and selection criteria are documented in
Section 3 of the report. No personally identifiable information beyond
self-chosen Twitter handles is retained, and all analysis is on aggregate
patterns.
