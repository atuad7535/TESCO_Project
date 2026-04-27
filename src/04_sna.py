"""
Stage 4 — Social Network Analysis on the Tesco Twitter graph.

We build a directed interaction graph from three edge types:
    * reply  : author -> in_reply_to_screen_name
    * retweet: author -> retweeted_user (screen_name)
    * mention: author -> mentioned_user (entities.user_mentions)

Edges are weighted by frequency. We then compute:
    * basic structural statistics (n, m, density, average clustering)
    * degree, in-degree, out-degree centrality (top 20)
    * PageRank (top 20)
    * Louvain community detection on the undirected projection

Input:  data/tweets_sentiment.parquet  (we want sentiment to colour communities)
Output: data/graph_summary.txt
        data/centrality_top.csv
        data/communities.csv
        data/edges.parquet
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import community as community_louvain  # python-louvain
import networkx as nx
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

MIN_EDGE_WEIGHT = 2  # drop singleton edges before community detection


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of (source, target, type) edges."""
    edges: list[tuple[str, str, str]] = []
    for row in df.itertuples(index=False):
        src = getattr(row, "author_screen_name")
        if not src or not isinstance(src, str):
            continue
        # reply
        tgt = getattr(row, "in_reply_to_screen_name", None)
        if isinstance(tgt, str) and tgt and tgt != src:
            edges.append((src, tgt, "reply"))
        # mentions (parquet -> may be ndarray)
        mentions = getattr(row, "mentions", None)
        if mentions is None:
            continue
        try:
            iter_m = list(mentions)
        except TypeError:
            continue
        for m in iter_m:
            if isinstance(m, str) and m and m != src:
                edges.append((src, m, "mention"))
    return pd.DataFrame(edges, columns=["source", "target", "type"])


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "tweets_sentiment.parquet")
    print(f"[load] {len(df):,} tweets")

    edges = build_edges(df)
    print(f"[edges] raw edge tokens = {len(edges):,}")

    weighted = (
        edges.groupby(["source", "target"]).size().reset_index(name="weight")
    )
    weighted.to_parquet(DATA_DIR / "edges.parquet", index=False)
    print(f"[edges] unique directed edges = {len(weighted):,}")

    # ------------------------------------------------------------------
    # Directed graph for centrality
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    for s, t, w in weighted.itertuples(index=False):
        G.add_edge(s, t, weight=int(w))
    print(f"[graph] directed: |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")

    # Density / reciprocity (cheap O(m))
    density = nx.density(G)
    try:
        reciprocity = nx.overall_reciprocity(G)
    except Exception:
        reciprocity = float("nan")

    # In/Out/Total degree
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}

    # PageRank (use weighted edges)
    print("[graph] computing PageRank ...")
    pr = nx.pagerank(G, weight="weight", alpha=0.85)

    cent_df = (
        pd.DataFrame({
            "user": list(G.nodes()),
            "in_degree": [in_deg.get(n, 0) for n in G.nodes()],
            "out_degree": [out_deg.get(n, 0) for n in G.nodes()],
            "total_degree": [total_deg.get(n, 0) for n in G.nodes()],
            "pagerank": [pr.get(n, 0.0) for n in G.nodes()],
        })
        .sort_values("pagerank", ascending=False)
        .reset_index(drop=True)
    )
    cent_df.head(50).to_csv(DATA_DIR / "centrality_top.csv", index=False)

    # ------------------------------------------------------------------
    # Community detection on the undirected, weight-filtered projection
    # ------------------------------------------------------------------
    UG = nx.Graph()
    filt = weighted[weighted["weight"] >= MIN_EDGE_WEIGHT]
    print(f"[comm] edges with weight>={MIN_EDGE_WEIGHT}: {len(filt):,}")
    for s, t, w in filt.itertuples(index=False):
        if UG.has_edge(s, t):
            UG[s][t]["weight"] += int(w)
        else:
            UG.add_edge(s, t, weight=int(w))

    # Largest connected component for stable community detection
    if UG.number_of_nodes() == 0:
        print("[comm] no edges left after filter — skipping community detection")
        partition = {}
    else:
        components = list(nx.connected_components(UG))
        giant = max(components, key=len)
        UG_giant = UG.subgraph(giant).copy()
        print(
            f"[comm] giant component: |V|={UG_giant.number_of_nodes():,}  "
            f"|E|={UG_giant.number_of_edges():,}"
        )
        partition = community_louvain.best_partition(UG_giant, weight="weight", random_state=42)

    if partition:
        modularity = community_louvain.modularity(partition, UG_giant, weight="weight")
        comm_counts = Counter(partition.values())
        print(f"[comm] {len(comm_counts)} communities, modularity = {modularity:.4f}")
        comm_df = pd.DataFrame(
            [{"user": u, "community": c} for u, c in partition.items()]
        )
        # join sentiment of authors' own tweets to colour communities
        author_sent = (
            df.groupby("author_screen_name")["sentiment_score"].mean().rename("avg_sent")
        )
        comm_df = comm_df.merge(author_sent, left_on="user", right_index=True, how="left")
        comm_df.to_csv(DATA_DIR / "communities.csv", index=False)
    else:
        modularity = float("nan")
        comm_counts = Counter()

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    n_comm = len(comm_counts)
    largest = comm_counts.most_common(5)
    summary = (
        f"Tesco Twitter interaction graph — summary\n"
        f"=========================================\n"
        f"Nodes (users)          : {G.number_of_nodes():,}\n"
        f"Directed edges         : {G.number_of_edges():,}\n"
        f"Density                : {density:.6f}\n"
        f"Reciprocity            : {reciprocity:.4f}\n"
        f"Edges w/ weight>={MIN_EDGE_WEIGHT}    : {len(filt):,}\n"
        f"Communities (Louvain)  : {n_comm}\n"
        f"Modularity             : {modularity:.4f}\n"
        f"\n"
        f"Top 5 communities by size:\n"
    )
    for cid, sz in largest:
        summary += f"  community {cid}: {sz:,} members\n"
    summary += "\nTop 10 PageRank users:\n"
    for _, r in cent_df.head(10).iterrows():
        summary += (
            f"  @{r['user']:<22}  pagerank={r['pagerank']:.5f}  "
            f"in={r['in_degree']:>6}  out={r['out_degree']:>6}\n"
        )

    (DATA_DIR / "graph_summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
