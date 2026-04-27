# Customer Voice in a Crisis Year: A Mixed-Method Twitter Analysis of Tesco During 2020

**Unit:** EFIMM0139 — Social Media and Web Analytics
**Word count (main body, excluding references and appendices):** ~3,000

## Abstract

This report examines what UK consumers and campaigners said about Tesco on Twitter during 2020, the year COVID-19 reshaped grocery retail. A pipeline combining transformer-based sentiment classification (cardiffnlp/twitter-roberta-base-sentiment-latest), Latent Dirichlet Allocation (K = 12, c_v = 0.32) on a high-confidence negative sub-corpus, and a directed mention/reply graph (26,781 nodes, 52,040 edges, modularity 0.65) is applied to 92,732 English-language tweets. The sentiment split is much flatter than the J-shaped curve typical of product reviews: 33.4% negative, 36.9% neutral, 29.7% positive. Topic modelling shows that organised activism — not individual customer complaints — accounts for roughly half of negative content, with animal-welfare campaigns alone representing about 30% of high-confidence negatives. Event-window comparisons reveal a 12-percentage-point increase in share-negative around the 23 March 2020 lockdown announcement, and the largest single positive shift in the sample (+0.69 mean score) around the first lockdown easing in June. The findings argue for treating Twitter sentiment as a leading indicator of *campaign* exposure as much as customer experience, and for communicating operational decisions (such as cuts to in-store cleaning) more carefully during crises.

---

## 1. Introduction

The grocery sector was one of the few categories that remained physically open through the United Kingdom's first national lockdown. Tesco, with roughly 27% of the UK grocery market in 2020 (Kantar, 2020), found itself central to a public conversation that was simultaneously about food security, public-health compliance, and the ethics of large companies receiving state support. Twitter was, for much of the year, the closest thing to a real-time complaint book for the sector — and it remains a useful place to study consumer voice because tweets are public, dated, and networked through replies, retweets and mentions (Russell, 2013).

This report has three motivating questions. First, what topics drove negative discussion of Tesco in 2020, and how do those topics relate to documented operational and reputational events? Second, how does a transformer sentiment classifier compare with the lexicon-based VADER baseline on this kind of brand-tagged Twitter data? Third, what does the structure of the conversation graph tell us about whose voice dominates — individual customers, competitors, or organised campaigning groups?

The contribution is largely empirical. I apply three techniques from the unit — sentiment analysis, topic modelling and social-network analysis — to a 96,702-tweet, year-long corpus, and link the results to verified UK retail and policy events. Section 2 reviews the relevant literature. Section 3 describes the data and methods. Section 4 presents the four sets of findings: the validated sentiment split, the topic structure of negative discussion, the network structure, and the event-window analysis. Section 5 draws three managerial recommendations and notes the main limitations.

## 2. Literature Review

This study connects three strands of research.

**Digital consumer voice as a complement to traditional research.** Survey and focus-group methods remain useful but are slow, expensive, and exposed to social-desirability bias (Paulhus, 1991). Humphreys and Wang (2018), in their *Journal of Consumer Research* survey of automated text analysis, argue that voluntary social-media expression captures phenomena traditional instruments miss — particularly emotional intensity and rapidly evolving issues. Twitter has, however, well-known representativeness problems: only a minority of the population tweets, and that minority skews younger and more politically engaged (Mellon and Prosser, 2017). Schoenmueller, Netzer and Stahl (2020), writing in the *Journal of Marketing Research*, show that the polarity distribution of online opinions varies by category and platform; for branded discussion on Twitter the visible signal is biased toward complaint and campaigning rather than toward routine satisfaction.

**NLP for sentiment and topic discovery.** Lexicon-based methods such as VADER (Hutto and Gilbert, 2014) provide a calibrated, transparent baseline but struggle with negation, sarcasm and context-dependent terms. The pretrained-then-fine-tuned transformer paradigm introduced by Devlin et al. (2019) for BERT has substantially improved sentiment performance on noisy text. For Twitter specifically, Loureiro et al. (2022) released a RoBERTa model fine-tuned on roughly 124 million tweets, which has become a common choice for tweet-level sentiment in the management literature; Choudhury et al. (2019), in *Strategic Management Journal*, demonstrate the broader managerial value of machine-learning-based text analysis. For topic discovery, Latent Dirichlet Allocation (Blei, Ng and Jordan, 2003) remains the most widely used method, and Sia, Dalmia and Mielke (2020) show that filtering by sentiment before topic modelling sharpens the resulting topics — the design choice followed in Section 3 below. Röder, Both and Hinneburg (2015) provide the c_v coherence metric used here for K selection.

**Network structure of brand discussion.** Marin and Wellman's (2011) handbook chapter introduces the methodological vocabulary — nodes, ties, density, centrality and community detection — that frames most empirical SNA work. For brand-tagged Twitter discussion, the literature documents that activist and campaigning accounts often act as super-spreaders, producing audience cascades that are quantitatively distinguishable from organic customer chatter (Bruns and Stieglitz, 2013). Louvain modularity optimisation (Blondel et al., 2008) is a standard technique for surfacing such clusters and is used here. Wu (2023), in *Management Science*, shows that text-derived signal can predict supply-chain risk; the analogue here is that text-derived signal can flag *reputational* risk, particularly when a topic is structurally dominated by a small set of campaign accounts.

Across these strands the methodological gap this report addresses is small but specific: most published applications focus on either the customer-experience or the campaigning side of brand Twitter, but not on the cross-cutting question of how the two can be separated within a single corpus.

## 3. Methodology and Data

### 3.1 Research design

The unit of analysis is the individual tweet; the empirical strategy combines three techniques (sentiment, topic, network) on the same corpus so that results from each can be cross-referenced. The primary outputs are (a) a class label and confidence score per tweet, (b) a dominant LDA topic for each high-confidence negative tweet, and (c) a centrality and community membership for each user. The temporal layer aggregates tweet-level sentiment to monthly and daily series for the event-window comparison.

### 3.2 Dataset and ethics

The corpus is a publicly-shared academic dataset of UK supermarket tweets (`tesco.json`, 489 MB, JSON in pandas column-oriented form) made available through the unit's shared Drive folder. Per the unit handbook, this counts as an *established dataset* rather than a self-collected one; I acknowledge that this limits the data-collection mark, but the trade-off was deliberate, given the time-window of the assignment and the size and schema-completeness of the dataset (96,702 records with full Twitter v1.1 entities and reply/retweet metadata, suitable for SNA without further enrichment).

Three ethical considerations apply. The tweets are public when posted, but Twitter's terms of service and the British Psychological Society's (2021) guidance on internet-mediated research both stress that aggregation rather than re-publication is the appropriate analytic strategy; no individual tweet is reproduced verbatim in the body of this report, and direct quotations in the figures are limited to public accounts of large organisations (e.g. brands, broadcasters, registered campaigning groups). User identifiers are retained only at handle level, which is the same level of identification under which the content was originally published. No location or follower-graph crawling beyond what is in the dataset was performed.

### 3.3 Cleaning and preparation

The raw file decomposes into 37 columns with 96,702 rows. The `created_at` field, encoded in epoch milliseconds, was cast to UTC timestamps and the corpus checked to span 1 January 2020 to 30 December 2020. Cleaning then proceeded in four steps. First, non-English tweets were dropped using Twitter's own `lang` field, which removed 3,970 records (largely Spanish, Tagalog and undefined). Second, the visible `text` field was replaced by `extended_tweet.full_text` where the original tweet was truncated, so that downstream analysis sees the full tweet rather than the 140-character preview. Third, URLs were stripped and whitespace was collapsed; @-mentions and # were preserved at this stage because the SNA pipeline relies on them. Finally, exact duplicates by `id_str` were removed. The cleaned corpus contains 92,732 records: 58,121 original tweets, 34,611 retweets, 36,338 replies, and 4,744 quote-tweets. Type counts overlap because a tweet can be both a reply and contain a quote.

### 3.4 Analytical techniques

**Sentiment classification.** I use the `cardiffnlp/twitter-roberta-base-sentiment-latest` model (Loureiro et al., 2022). The choice over VADER (Hutto and Gilbert, 2014) and vanilla BERT (Devlin et al., 2019) is justified on three grounds: (a) the model's pretraining corpus is roughly 124 million tweets, so its tokenisation and embedding space are domain-matched; (b) it returns three-class probabilities (negative / neutral / positive) which align with the LDA-filtering and event-study designs; and (c) on independent benchmarks it outperforms both VADER and base-BERT on Twitter-specific evaluation sets. Inference was run on Apple-Silicon MPS, batch size 64, max length 128 tokens. To address the criticism that a single transformer judgement should not be taken at face value, I additionally ran VADER over the same corpus and computed Cohen's kappa and a 3 × 3 agreement matrix between the two models. The transformer's confidence (the softmax probability of its top class) is also retained and used downstream as a filter for the topic-modelling stage.

**Topic modelling.** Following Sia, Dalmia and Mielke (2020), LDA is applied not to the whole corpus but to the high-confidence negative sub-corpus (RoBERTa label = negative AND confidence ≥ 0.80, n = 14,767 before token-filter, n = 13,014 after). Pre-processing applied lower-casing, removal of @-mentions and #, an English-stopword list extended with a small domain-specific blocklist (e.g. "tesco", "tescos", common contractions and chat fillers), tokenisation by regex and WordNet lemmatisation. The Gensim dictionary was filtered to terms appearing in at least 10 documents and in fewer than 50% of documents. K was selected by maximising c_v coherence (Röder, Both and Hinneburg, 2015) over K ∈ {5, 7, 10, 12}; coherence rose monotonically and K = 12 was retained (c_v = 0.32). An interactive pyLDAvis dashboard was generated and is referenced in Appendix B.

**Social network analysis.** A directed graph was constructed with one node per user (`screen_name`) and edges for replies (author → in_reply_to_screen_name) and mentions (author → mentioned screen name). Self-edges were removed. Edges were weighted by frequency. PageRank (Brin and Page, 1998), implemented in NetworkX with α = 0.85 and edge weights, was used as the principal centrality measure because it discounts incoming links from low-importance accounts. For community detection, the directed graph was projected to undirected, edges with weight < 2 were removed to suppress one-off interactions, and the Louvain algorithm (Blondel et al., 2008) — implemented in `python-louvain` — was applied to the giant connected component (8,508 nodes, 9,959 edges).

**Temporal and event-window analysis.** Tweet-level sentiment scores (defined as `P(positive) − P(negative)`, on [−1, 1]) were aggregated to daily and monthly means. A list of ten verified UK retail and policy events was assembled from contemporary news sources and the ONS coronavirus archive (Office for National Statistics, 2020). For each event, mean sentiment and share-negative were compared in the 14-day windows immediately before and after the event date. The comparison is associational; no causal claim is made.

## 4. Analysis and Results

### 4.1 Volume, types and overall sentiment

The cleaned corpus contains 92,732 English tweets distributed unevenly across 2020. Volume peaked in March (around 14,000 tweets in the lockdown month) and fell back to roughly 5–7,000 per month from June onwards (Figure 1, Appendix A). Reply tweets (39%) and retweets (37%) together account for most of the corpus; only 63% of the corpus is original content if quote-tweets are counted as semi-original.

The RoBERTa classifier returned 33.4% negative, 36.9% neutral and 29.7% positive (Figure 2). This is much flatter than the J-shaped distribution familiar from product reviews (Hu, Zhang and Pavlou, 2009), where roughly three-quarters of records sit at the highest rating. The difference is consistent with two structural features of Twitter as a data source. First, voluntary tweets are not tied to a transactional event the way reviews are, so the self-selection bias points different ways: reviewers tend to be satisfied buyers, whereas tweeters who tag a brand are disproportionately motivated by complaint, advocacy or campaigning. Second, the platform's reply-and-retweet architecture multiplies emotionally engaged content. The mean signed score in the corpus is −0.0065 — almost exactly neutral on aggregate — but this masks substantial month-to-month variation discussed in Section 4.5.

The VADER baseline is a useful sanity check. Across the same 92,732 tweets, VADER and RoBERTa agree on the class label only 62.4% of the time, with Cohen's kappa = 0.44 (moderate agreement on Landis and Koch's 1977 scale). The disagreement is asymmetric. VADER labels 50.4% of tweets positive against RoBERTa's 29.7%, mostly by re-classifying RoBERTa's *neutrals* and a non-trivial slice of its *negatives* as positive. Figure 10 shows the full confusion matrix. The pattern is consistent with the theoretical concern that a lexicon-based classifier with no explicit handling of negation, irony or compound sentence structure tends to be over-positive on tweets that contain otherwise positively-valenced product nouns ("delivery", "thanks", "free") sitting inside a complaint. RoBERTa is therefore retained as the primary classifier; VADER scores are kept in the dataset for transparency.

### 4.2 What people complained about: LDA topics

Twelve topics were extracted from the negative sub-corpus. Inspecting top-words and top-probability example tweets (sample given in Appendix A) suggests they collapse into six interpretable themes once near-duplicate animal-welfare topics are merged.

| Collapsed theme | Original LDA topics | Share of negatives |
|---|---|---|
| Animal welfare campaigns ("Frankenchickens", egg, dates) | 0, 1, 4, 7, 11 | ≈ 27% |
| In-store discrimination and racism allegations | 10 | 17% |
| Delivery-slot scarcity | 6 | 14% |
| State support, dividend and tax | 9 | 11% |
| Cleaning-staff cuts during pandemic | 2, 5 | 11% |
| Mask compliance and store-experience | 3, 8 | 19% |

Two findings stand out. First, **the largest single category is not a customer-experience theme but an organised campaign**: the Open Cages investigation into chicken farms supplying Tesco, which produced cascades of retweets in late 2020 and recurring activity throughout the year. Roughly one in four high-confidence negative tweets falls into this cluster. Second, **operational decisions can become reputational events very quickly**. Tesco's August 2020 plan to remove cleaning staff and reassign their work to shop-floor employees produced a discrete, identifiable sub-corpus (Topics 2 and 5, ≈11%) with vocabulary dominated by "axing", "thousand", "cleaner", "hygiene", and "petition" — language that mirrors the petition wording rather than diffuse customer complaint. The size of this cluster is striking given the action did not directly affect customer experience.

The smaller themes are recognisable: delivery-slot frustration concentrated in March–April, when Tesco prioritised vulnerable customers and the slot-booking system became contested; mask-compliance disputes concentrated in summer; and a sustained, lower-volume thread on Tesco's £585m business-rates relief and subsequent dividend, which the company later returned to government.

The Topic 8 cluster contains a non-trivial volume of profanity-heavy content; this is a known characteristic of brand-tagged Twitter and motivates the recommendation in Section 5 to keep automated sentiment dashboards behind moderation.

### 4.3 Who shapes the conversation: network structure

The mention-and-reply graph contains 26,781 distinct users and 52,040 directed weighted edges. Density is very low at 7.3 × 10⁻⁵ — typical of public Twitter graphs at this scale. PageRank concentrates sharply: @Tesco itself is the dominant hub (PR = 0.182, in-degree 25,210, out-degree 1 — an information-receiving rather than information-broadcasting account). The next four hubs are competitor brand accounts (@sainsburys, @Morrisons, @asda, @waitrose, @AldiUK), each receiving roughly 1,300–2,600 mention-edges, suggesting that comparative tweets ("Tesco vs Asda…") are a substantial slice of the corpus. The presence of @raceforlife — Cancer Research UK's running-race brand, of which Tesco is the long-running headline sponsor — at PR = 0.159 is a methodological reminder that PageRank can be inflated by very tight, low-volume affiliated networks.

Louvain detection on the weight-filtered giant component (8,508 nodes) returned 556 communities with modularity 0.65. A modularity above 0.4 is conventionally taken as evidence of meaningful community structure (Newman, 2006); the value here is high, implying that conversation around Tesco is genuinely divided rather than one undifferentiated stream. The largest community (4,297 members, 50% of the giant component) is the @Tesco-customer-service cluster — users complaining or asking questions and being answered by Tesco's official accounts. Smaller communities map onto recognisable groupings: a pets/lifestyle cluster, a campaigning cluster centred on welfare accounts, a journalism cluster around UK retail correspondents, and several local-store clusters (Figure 11).

### 4.4 Temporal sentiment and event windows

Monthly mean sentiment was strongly negative in early 2020, rose into clearly positive territory through summer as supply normalised, and returned to mild negativity in the autumn lockdown period (Figure 6). The 14-day pre/post comparison around ten anchor events (Figure 7, Table 2) makes three patterns visible. The 23 March 2020 lockdown announcement coincides with share-negative rising from 24.5% to 36.6% (Δ = +12.1 percentage points) — the largest negative move in the dataset, and a reasonable reflection of the panic-buying, queue and slot-availability complaints that dominated late March. The first lockdown easing on 15 June 2020 shows the largest *positive* movement (Δ mean = +0.69), although the very low pre-event volume (n = 2,402 tweets in the prior 14 days) inflates the magnitude of this comparison. The October 2020 tier announcement, similarly, shows a large positive shift (Δ = +0.28), perhaps a relief signal as harder restrictions were avoided.

Two negative findings are worth recording. First, the WHO pandemic declaration on 11 March was associated with a *positive* sentiment shift (Δ = +0.09), driven by solidarity-style tweets thanking key workers — a useful corrective to the assumption that public-health bad news translates linearly into brand-negative sentiment. Second, the Tier-4 announcement of 19 December produced almost no shift (Δ = −0.045), because by that point Twitter discussion was already saturated with seasonal and politicised content. None of these comparisons is causal; macro events overlap, holidays interfere with volume, and dataset artefacts (especially campaign-driven retweet bursts) affect both windows simultaneously.

## 5. Conclusion, Recommendations and Limitations

### 5.1 Summary

Across 92,732 Tesco-related tweets in 2020, sentiment was approximately balanced (33% negative, 37% neutral, 30% positive), much flatter than the J-shape typical of product reviews. Latent topic structure showed that organised campaigning — particularly on animal welfare — accounts for roughly half of negative discussion, with operational and policy themes (delivery slots, cleaning-staff cuts, business-rates relief) making up most of the rest. Network analysis showed a single dominant hub (@Tesco), a competitor cluster, and a sharply modular structure (modularity = 0.65) with a large customer-service core. Event-window analysis tied the largest negative shift to the 23 March lockdown and the largest positive shift to early-summer easing.

### 5.2 Recommendations

I would offer three practical recommendations to a Tesco social-listening team based on this analysis.

1. *Treat campaign discussion and customer discussion as two distinct queues.* Roughly half of high-confidence negative tweets are campaign-amplified rather than incident-driven. A single sentiment dashboard that mixes them will overstate operational problems and understate reputational ones. Splitting the queue (for example, by separating tweets whose top-three retweet path includes a campaigning account) would give an actionable view.
2. *Communicate operational decisions during a crisis with the public-facing implications already mapped.* The 2020 cleaning-staff decision generated a clearly identifiable negative cluster of comparable size to delivery-slot scarcity, despite the latter being a far larger operational issue. The asymmetry suggests that the *narrative legibility* of an operational decision matters more than its scale.
3. *Use share-negative, not mean sentiment, as the leading indicator during a crisis.* The mean sentiment series moves between roughly −0.2 and +0.5 across 2020 and is sensitive to volume. Share-negative is a more stable diagnostic and showed a 12-percentage-point shift around the lockdown announcement — the kind of signal that a service-recovery team can act on inside a 24-hour window.

### 5.3 Limitations

Three limitations are worth flagging. First, the dataset is pre-collected and the 2020 period is now history; the analysis is descriptive of a specific year and does not generalise to a steady state. Second, retweet inflation is real: a small number of campaign accounts produced a disproportionate share of negative content. Confidence-filtering and topic-grouping mitigate this but do not eliminate it. Third, all event-window comparisons are associational; macro-economic and seasonal confounds are present in every window. A causal design would require a control corpus from a comparable retailer, which is feasible (the project's wider Drive folder includes Sainsbury's and Waitrose data) but is left to further work.

---

## References

Blei, D.M., Ng, A.Y. and Jordan, M.I. (2003) 'Latent Dirichlet allocation', *Journal of Machine Learning Research*, 3, pp. 993–1022.

Blondel, V.D., Guillaume, J.-L., Lambiotte, R. and Lefebvre, E. (2008) 'Fast unfolding of communities in large networks', *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.

Brin, S. and Page, L. (1998) 'The anatomy of a large-scale hypertextual web search engine', *Computer Networks and ISDN Systems*, 30(1–7), pp. 107–117.

British Psychological Society (2021) *Ethics guidelines for internet-mediated research*. Leicester: BPS.

Bruns, A. and Stieglitz, S. (2013) 'Towards more systematic Twitter analysis: metrics for tweeting activities', *International Journal of Social Research Methodology*, 16(2), pp. 91–108.

Choudhury, P., Wang, D., Carlson, N.A. and Khanna, T. (2019) 'Machine learning approaches to facial and text analysis: discovering CEO oral communication styles', *Strategic Management Journal*, 40(11), pp. 1705–1732.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: pre-training of deep bidirectional transformers for language understanding', in *Proceedings of NAACL-HLT 2019*, pp. 4171–4186.

Hu, N., Zhang, J. and Pavlou, P.A. (2009) 'Overcoming the J-shaped distribution of product reviews', *Communications of the ACM*, 52(10), pp. 144–147.

Humphreys, A. and Wang, R.J.-H. (2018) 'Automated text analysis for consumer research', *Journal of Consumer Research*, 44(6), pp. 1274–1306.

Hutto, C.J. and Gilbert, E. (2014) 'VADER: a parsimonious rule-based model for sentiment analysis of social media text', in *Proceedings of the 8th International AAAI Conference on Web and Social Media (ICWSM 2014)*, Ann Arbor, MI, pp. 216–225.

Kantar (2020) *Grocery market share — UK*. London: Kantar Worldpanel. Available at: https://www.kantarworldpanel.com/global/grocery-market-share/great-britain (Accessed: 26 April 2026).

Landis, J.R. and Koch, G.G. (1977) 'The measurement of observer agreement for categorical data', *Biometrics*, 33(1), pp. 159–174.

Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L. and Camacho-Collados, J. (2022) 'TimeLMs: diachronic language models from Twitter', in *Proceedings of ACL 2022: System Demonstrations*, pp. 251–260.

Marin, A. and Wellman, B. (2011) 'Social network analysis: an introduction', in Scott, J. and Carrington, P.J. (eds.) *The SAGE Handbook of Social Network Analysis*. London: SAGE, pp. 11–25.

Mellon, J. and Prosser, C. (2017) 'Twitter and Facebook are not representative of the general population: political attitudes and demographics of British social media users', *Research and Politics*, 4(3), pp. 1–9.

Newman, M.E.J. (2006) 'Modularity and community structure in networks', *Proceedings of the National Academy of Sciences*, 103(23), pp. 8577–8582.

Office for National Statistics (2020) *Coronavirus and the impact on UK retail*. Newport: ONS. Available at: https://www.ons.gov.uk/economy/inflationandpriceindices/articles/coronavirusandtheimpactonukretail (Accessed: 26 April 2026).

Pang, B. and Lee, L. (2008) 'Opinion mining and sentiment analysis', *Foundations and Trends in Information Retrieval*, 2(1–2), pp. 1–135.

Paulhus, D.L. (1991) 'Measurement and control of response bias', in Robinson, J.P., Shaver, P.R. and Wrightsman, L.S. (eds.) *Measures of Personality and Social Psychological Attitudes*. San Diego: Academic Press, pp. 17–59.

Röder, M., Both, A. and Hinneburg, A. (2015) 'Exploring the space of topic coherence measures', in *Proceedings of WSDM 2015*, pp. 399–408.

Russell, M.A. (2013) *Mining the Social Web: Data Mining Facebook, Twitter, LinkedIn, Google+, GitHub, and More*. 2nd edn. Sebastopol, CA: O'Reilly Media.

Schoenmueller, V., Netzer, O. and Stahl, F. (2020) 'The polarity of online reviews', *Journal of Marketing Research*, 57(5), pp. 853–877.

Sia, S., Dalmia, A. and Mielke, S.J. (2020) 'Tired of topic models? Clusters of pretrained word embeddings make for fast and good topics too!', in *Proceedings of EMNLP 2020*, pp. 1728–1736.

Wu, D. (2023) 'Text-based measure of supply chain risk exposure', *Management Science*, forthcoming.

---

## Appendix A — Additional Figures

Figures referenced in the main text are reproduced at higher resolution below. All figures are generated by `src/06_make_figures.py` from the artefacts written by Stages 1–5 of the pipeline.

- Figure 1. Daily volume of Tesco-related English tweets, 2020. (`figures/f01_volume_by_day.png`)
- Figure 2. RoBERTa sentiment-class distribution. (`figures/f02_sentiment_distribution.png`)
- Figure 3. LDA c_v coherence over K ∈ {5, 7, 10, 12}. (`figures/f03_lda_coherence.png`)
- Figure 4. Top 15 words per LDA topic. (`figures/f04_topic_top_words.png`)
- Figure 5. Topic share of the high-confidence negative corpus. (`figures/f05_topic_share.png`)
- Figure 6. Monthly mean sentiment and share negative. (`figures/f06_monthly_sentiment.png`)
- Figure 7. Event-window 14-day pre/post deltas. (`figures/f07_event_study.png`)
- Figure 8. Top 20 users by PageRank. (`figures/f08_pagerank_top.png`)
- Figure 9. Top 10 Louvain communities, sized by membership and coloured by mean author sentiment. (`figures/f09_community_sentiment.png`)
- Figure 10. RoBERTa vs VADER agreement matrix. (`figures/f10_model_agreement.png`)
- Figure 11. Spring-layout visualisation of the top-400-node sub-graph, coloured by Louvain community. (`figures/f11_graph_viz.png`)

## Appendix B — Interactive LDA Dashboard

A pyLDAvis-generated interactive dashboard is included with the submission as `figures/lda_interactive.html`. It shows the inter-topic distance map (multidimensional scaling) on the left and a relevance-controlled per-topic word ranking on the right. Recommended viewing setting: λ ≈ 0.6 (Sievert and Shirley, 2014).

## Appendix C — Reproducibility

The full pipeline is the seven scripts under `src/`:

1. `01_load_clean.py` — JSON → cleaned parquet
2. `02_sentiment.py` — RoBERTa sentiment over the corpus
3. `02b_vader_baseline.py` — VADER baseline and agreement matrix
4. `03_topic_lda.py` — LDA with c_v K-selection on confidence-filtered negatives
5. `04_sna.py` — NetworkX directed-graph statistics, PageRank and Louvain
6. `05_temporal.py` — daily/monthly aggregation and event-window comparison
7. `06_make_figures.py` — all PNG figures used in the report

Reproduction:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
python src/01_load_clean.py --input ~/Downloads/tesco.json
python src/02_sentiment.py
python src/02b_vader_baseline.py
python src/03_topic_lda.py
python src/04_sna.py
python src/05_temporal.py
python src/06_make_figures.py
```

Total runtime on a 2024 MacBook Air (M3, 16 GB) with MPS: approximately 12 minutes, dominated by the RoBERTa pass (≈ 7 minutes) and LDA training (≈ 30 seconds).

## Appendix D — Source Code

The full source code of all seven scripts is reproduced verbatim below for the assignment requirement that the analytical pipeline be auditable.

> *(The seven scripts in the `src/` folder are inserted here in pipeline order at submission time. Each script is preceded by its filename. The total source listing is approximately 530 lines.)*
