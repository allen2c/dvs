import logging
import pathlib
import typing

if typing.TYPE_CHECKING:
    import networkx as nx


logger = logging.getLogger(__name__)


def export_graph_png(
    G: "nx.DiGraph",
    output_path: pathlib.Path | str,
) -> None:

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
    except ImportError:
        logger.warning("Matplotlib not found. Skipping PNG export.")
        return

    # Prefer a CJK-capable font so Chinese is rendered correctly in PNG output
    try:
        preferred_fonts: list[str] = [
            "PingFang SC",  # macOS
            "Hiragino Sans GB",
            "Heiti SC",
            "Heiti TC",
            "Songti SC",
            "STSong",
            "STHeiti",
            "Microsoft YaHei",  # Windows
            "SimHei",
            "Noto Sans CJK SC",  # Noto/Source Han
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "Source Han Sans CN",
            "Source Han Sans SC",
            "WenQuanYi Micro Hei",  # Linux
            "Arial Unicode MS",
        ]
        installed_fonts: typing.Set[str] = {f.name for f in fm.fontManager.ttflist}
        chosen_font: str | None = next(
            (f for f in preferred_fonts if f in installed_fonts), None
        )
        if chosen_font is not None:
            mpl.rcParams["font.family"] = chosen_font
            mpl.rcParams["font.sans-serif"] = [chosen_font]
            mpl.rcParams["axes.unicode_minus"] = False
            logger.info(f"Using font family for CJK: [green]{chosen_font}[/green]")
        else:
            logger.warning("No preferred CJK font found. Chinese text may not render.")
    except Exception as e:
        logger.error(f"Font configuration failed: {e}")

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#e1f5fe", alpha=0.9)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        edge_color="#b0bec5",
        width=1.5,
        arrows=True,
        arrowsize=20,
    )
    label_font_family: str = (
        chosen_font
        if "chosen_font" in locals() and chosen_font is not None
        else "sans-serif"
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_family=label_font_family)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#e53935",
        font_family=label_font_family,
    )

    plt.title("Knowledge Graph", size=20)
    plt.tight_layout()
    plt.savefig(output_path, format="PNG", dpi=300)
    plt.close()

    logger.info(f"Graph PNG saved to [green]{output_path}[/green].")

    return None
