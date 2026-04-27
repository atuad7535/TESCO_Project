"""
Stage 7 — assemble the final .docx.

We expand the report.md into a "report_full.md" by:
  - replacing the Appendix-A figure list with actual ![](path) markdown so
    pandoc embeds the PNGs;
  - replacing the Appendix-D placeholder block with the verbatim source of
    every Python file in src/ (pipeline order), wrapped in fenced
    code-blocks so pandoc renders them as monospace.

Then we run pandoc to produce report.docx.
"""
import re
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT / "src"
FIG_DIR = PROJECT / "figures"


SCRIPT_ORDER = [
    "01_load_clean.py",
    "02_sentiment.py",
    "02b_vader_baseline.py",
    "03_topic_lda.py",
    "04_sna.py",
    "05_temporal.py",
    "06_make_figures.py",
]


def expand_figures(md: str) -> str:
    """Replace the bullet list of figure references in Appendix A with
    actual markdown image embeds + captions."""
    # Find the appendix-A figure block bullet list and rewrite to img embeds.
    fig_block = []
    figs = [
        ("f01_volume_by_day.png", "Figure 1. Daily volume of Tesco-related English tweets, 2020."),
        ("f02_sentiment_distribution.png", "Figure 2. RoBERTa sentiment-class distribution (n = 92,732)."),
        ("f03_lda_coherence.png", "Figure 3. c_v coherence across K ∈ {5, 7, 10, 12}; K = 12 selected."),
        ("f04_topic_top_words.png", "Figure 4. Top 15 words per LDA topic."),
        ("f05_topic_share.png", "Figure 5. Share of the high-confidence negative corpus per topic."),
        ("f06_monthly_sentiment.png", "Figure 6. Monthly mean sentiment and share negative, 2020."),
        ("f07_event_study.png", "Figure 7. Event-window 14-day pre/post sentiment deltas (associational)."),
        ("f08_pagerank_top.png", "Figure 8. Top 20 users by PageRank in the Tesco interaction graph."),
        ("f09_community_sentiment.png", "Figure 9. Top 10 Louvain communities, by membership and mean author sentiment."),
        ("f10_model_agreement.png", "Figure 10. RoBERTa vs VADER label-agreement matrix."),
        ("f11_graph_viz.png", "Figure 11. Spring-layout sub-graph of the top-400 nodes, coloured by Louvain community."),
    ]
    for fname, caption in figs:
        path = FIG_DIR / fname
        if not path.exists():
            continue
        fig_block.append(f"\n![{caption}]({path.as_posix()})\n\n*{caption}*\n")

    # Replace the bullet list in appendix A
    pattern = re.compile(
        r"(## Appendix A — Additional Figures\n\nFigures referenced.*?from the artefacts written by Stages 1–5 of the pipeline\.\n\n)(- Figure 1\..*?\(`figures/f11_graph_viz\.png`\)\n)",
        re.S,
    )
    figs_block_str = "".join(fig_block)
    new_md, count = pattern.subn(lambda m: m.group(1) + figs_block_str + "\n", md)
    if count == 0:
        # fallback — append at end of appendix A header if pattern doesn't match
        new_md = md + "\n" + "".join(fig_block)
    return new_md


def expand_source_appendix(md: str) -> str:
    blocks = []
    for fname in SCRIPT_ORDER:
        path = SRC_DIR / fname
        if not path.exists():
            continue
        body = path.read_text()
        blocks.append(f"\n### `src/{fname}`\n\n```python\n{body}\n```\n")
    code_md = "".join(blocks)
    placeholder = re.compile(
        r"> \*\(The seven scripts in the `src/` folder.*?approximately 530 lines\.\)\*",
        re.S,
    )
    return placeholder.sub(lambda _m: code_md, md)


def main() -> None:
    md = (PROJECT / "report.md").read_text()
    md = expand_figures(md)
    md = expand_source_appendix(md)

    out_md = PROJECT / "report_full.md"
    out_md.write_text(md)
    print(f"[md] wrote {out_md} ({len(md):,} chars)")

    out_docx = PROJECT / "report.docx"
    cmd = [
        "pandoc",
        str(out_md),
        "-o",
        str(out_docx),
        "--from=markdown",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--highlight-style=tango",
    ]
    print("[pandoc]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[ok] wrote {out_docx}")


if __name__ == "__main__":
    main()
