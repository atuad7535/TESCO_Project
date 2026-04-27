"""
Stage 6 — Generate all PNG figures referenced in the report.

Reads the artefacts produced by stages 1–5 and writes high-resolution
PNGs to figures/. Each figure is captioned in the report.

Figures produced:
    f01_volume_by_day.png      — tweet volume over 2020
    f02_sentiment_distribution — overall sentiment class distribution
    f03_lda_coherence.png      — c_v across K candidates
    f04_topic_top_words.png    — top words per topic (heatmap-style)
    f05_topic_share.png        — topic share of negative corpus
    f06_monthly_sentiment.png  — monthly mean sentiment + share negative
    f07_event_study.png        — pre/post Δ for each event
    f08_pagerank_top.png       — top-20 PageRank users
    f09_community_sentiment    — average sentiment per community (top 10)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def fig01_volume() -> None:
    daily = pd.read_csv(DATA_DIR / "daily_sentiment.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(daily["date"], daily["n"], alpha=0.6, color="#1f77b4")
    ax.set_title("Daily volume of Tesco-related English tweets, 2020")
    ax.set_ylabel("Tweets per day")
    ax.set_xlabel("")
    fig.savefig(FIG_DIR / "f01_volume_by_day.png")
    plt.close(fig)


def fig02_sentiment_distribution() -> None:
    df = pd.read_parquet(DATA_DIR / "tweets_sentiment.parquet")
    counts = df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"])
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(
        counts.index, counts.values,
        color=["#d62728", "#7f7f7f", "#2ca02c"],
    )
    for b, p in zip(bars, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sentiment-class distribution (RoBERTa, n = {:,})".format(int(counts.sum())))
    ax.set_ylabel("Tweets")
    fig.savefig(FIG_DIR / "f02_sentiment_distribution.png")
    plt.close(fig)


def fig03_coherence() -> None:
    df = pd.read_csv(DATA_DIR / "lda_coherence.csv")
    best = df.loc[df["c_v"].idxmax()]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(df["k"], df["c_v"], "o-", color="#1f77b4")
    ax.scatter([best["k"]], [best["c_v"]], color="#d62728", zorder=5,
               label=f"selected K = {int(best['k'])}")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel(r"$c_v$ coherence")
    ax.set_title("LDA topic-coherence selection")
    ax.legend()
    fig.savefig(FIG_DIR / "f03_lda_coherence.png")
    plt.close(fig)


def fig04_top_words() -> None:
    tw = pd.read_csv(DATA_DIR / "lda_topics_top_words.csv")
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * len(tw)))
    ax.axis("off")
    rows = []
    for _, r in tw.iterrows():
        rows.append([f"Topic {int(r['topic'])}", r["top_words"]])
    tbl = ax.table(cellText=rows, colLabels=["", "Top 15 words (descending β)"],
                   cellLoc="left", loc="center", colWidths=[0.12, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("LDA topics — top words per topic", pad=10)
    fig.savefig(FIG_DIR / "f04_topic_top_words.png")
    plt.close(fig)


def fig05_topic_share() -> None:
    tp = pd.read_parquet(DATA_DIR / "lda_topics.parquet")
    share = tp["dominant_topic"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(share.index.astype(str), share.values * 100, color="#9467bd")
    for b, v in zip(bars, share.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Share of negative corpus (%)")
    ax.set_title("LDA topic share of high-confidence negative tweets")
    fig.savefig(FIG_DIR / "f05_topic_share.png")
    plt.close(fig)


def fig06_monthly_sentiment() -> None:
    monthly = pd.read_csv(DATA_DIR / "monthly_sentiment.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(monthly["month"], monthly["mean_score"], "o-", color="#1f77b4")
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mean sentiment score")
    axes[0].set_title("Monthly mean sentiment and share-negative, Tesco tweets 2020")
    axes[1].plot(monthly["month"], monthly["share_neg"] * 100, "s-", color="#d62728")
    axes[1].set_ylabel("Share negative (%)")
    axes[1].set_xlabel("Month")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.savefig(FIG_DIR / "f06_monthly_sentiment.png")
    plt.close(fig)


def fig07_event_study() -> None:
    ev = pd.read_csv(DATA_DIR / "event_study.csv")
    ev = ev.dropna(subset=["delta"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in ev["delta"]]
    bars = ax.barh(ev["event"], ev["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    for b, d in zip(bars, ev["delta"]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_xlabel(r"$\Delta$ mean sentiment (post 14d − pre 14d)")
    ax.set_title("Event study: 14-day pre/post sentiment shifts (associational)")
    ax.invert_yaxis()
    fig.savefig(FIG_DIR / "f07_event_study.png")
    plt.close(fig)


def fig08_pagerank() -> None:
    cent = pd.read_csv(DATA_DIR / "centrality_top.csv").head(20)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(cent["user"][::-1], cent["pagerank"][::-1], color="#17becf")
    ax.set_xlabel("PageRank")
    ax.set_title("Top 20 users by PageRank in Tesco interaction graph")
    fig.savefig(FIG_DIR / "f08_pagerank_top.png")
    plt.close(fig)


def fig09b_model_agreement() -> None:
    path = DATA_DIR / "model_agreement.csv"
    if not path.exists():
        return
    cm = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("RoBERTa vs VADER — agreement matrix")
    ax.set_xlabel("VADER label")
    ax.set_ylabel("RoBERTa label")
    fig.savefig(FIG_DIR / "f10_model_agreement.png")
    plt.close(fig)


def fig09c_graph_viz() -> None:
    """Compact spring layout of the top-N most-connected nodes coloured by community."""
    import networkx as nx
    edges_path = DATA_DIR / "edges.parquet"
    comm_path = DATA_DIR / "communities.csv"
    if not (edges_path.exists() and comm_path.exists()):
        return
    edges = pd.read_parquet(edges_path)
    comms = pd.read_csv(comm_path)
    # take the giant component edges by intersection with community members
    top_users = set(comms["user"].head(400))
    sub = edges[edges["source"].isin(top_users) & edges["target"].isin(top_users)]
    if sub.empty:
        return
    G = nx.Graph()
    for s, t, w in sub.itertuples(index=False):
        if G.has_edge(s, t):
            G[s][t]["weight"] += int(w)
        else:
            G.add_edge(s, t, weight=int(w))
    # restrict to top community labels for colouring
    comm_lookup = dict(zip(comms["user"], comms["community"]))
    top_comms = comms["community"].value_counts().head(8).index.tolist()
    palette = sns.color_palette("tab10", n_colors=len(top_comms))
    node_colours = []
    for n in G.nodes():
        c = comm_lookup.get(n)
        node_colours.append(palette[top_comms.index(c)] if c in top_comms else "#cccccc")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=120, weight=None)
    nx.draw_networkx_edges(G, pos, alpha=0.10, width=0.3, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_size=[15 + G.degree(n) * 0.9 for n in G.nodes()],
        node_color=node_colours, alpha=0.85, ax=ax, linewidths=0,
    )

    # top-10 handles to label, but skip @Tesco (it's the obvious centre hub)
    # so we don't waste a label on the most cluttered position
    deg_sorted = sorted(G.degree, key=lambda x: -x[1])
    label_targets = [n for n, _ in deg_sorted if n.lower() != "tesco"][:9]

    # spread labels around the boundary of the figure with leader lines:
    # for each handle, place the text at a constant radius from the centre,
    # rotating around 360 degrees, then draw a thin line from the node
    import math
    cx, cy = 0.0, 0.0
    angles = [i * 2 * math.pi / len(label_targets) - math.pi / 2 for i in range(len(label_targets))]
    R = 1.15
    for ang, name in zip(angles, label_targets):
        nx_pos = pos[name]
        lx, ly = cx + R * math.cos(ang), cy + R * math.sin(ang)
        ax.plot([nx_pos[0], lx], [nx_pos[1], ly], color="#444", linewidth=0.5, alpha=0.5, zorder=1)
        ha = "left" if lx > cx else "right"
        ax.text(
            lx, ly, f"@{name}", fontsize=9, ha=ha, va="center",
            bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.2", linewidth=0.5),
            zorder=5,
        )
    # explicitly mark @Tesco at its own position with a clear callout
    if "Tesco" in pos:
        tx, ty = pos["Tesco"]
        ax.text(
            tx, ty + 0.05, "@Tesco", fontsize=11, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25", linewidth=0.8),
            zorder=6,
        )

    ax.set_axis_off()
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("Tesco interaction graph (top-400 nodes, Louvain communities)")
    fig.savefig(FIG_DIR / "f11_graph_viz.png")
    plt.close(fig)


def fig09_community_sentiment() -> None:
    path = DATA_DIR / "communities.csv"
    if not path.exists():
        return
    comm = pd.read_csv(path)
    sizes = comm.groupby("community").size().rename("size")
    sent = comm.groupby("community")["avg_sent"].mean()
    summary = pd.concat([sizes, sent], axis=1).sort_values("size", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    norm = plt.Normalize(-0.5, 0.5)
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(v)) if not np.isnan(v) else "lightgrey" for v in summary["avg_sent"]]
    bars = ax.bar(summary.index.astype(str), summary["size"], color=colors)
    for b, v in zip(bars, summary["avg_sent"]):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Members")
    ax.set_title("Top 10 Louvain communities — size and mean author sentiment")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Mean author sentiment")
    fig.savefig(FIG_DIR / "f09_community_sentiment.png")
    plt.close(fig)


def main() -> None:
    fig01_volume()
    fig02_sentiment_distribution()
    fig03_coherence()
    fig04_top_words()
    fig05_topic_share()
    fig06_monthly_sentiment()
    fig07_event_study()
    fig08_pagerank()
    fig09_community_sentiment()
    fig09b_model_agreement()
    fig09c_graph_viz()
    print("[ok] figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
