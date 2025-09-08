import asyncio
import json
import pathlib
import typing

import agents
import diskcache
import logging_bullet_train as lbt
import openai
import openai_embeddings_model as oai_emb_model
from aps_agent import APSAgent
from ner_agent import NerAgent
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
)
from openai_embeddings_model import ModelSettings as EmbeddingModelSettings
from rich.console import Console

import dvs
from dvs.types.document import Document
from dvs.types.edge import Edge
from dvs.types.graphrag_result import GraphRAGResult
from dvs.types.node import Node
from dvs.utils.build_graph_from_documents import build_graph_from_documents
from dvs.utils.dump_graph import dump_graph
from dvs.utils.load_documents_from_directory import load_documents_from_directory
from dvs.utils.loads_graph import load_graph
from dvs.utils.query_expander import build_llm_query_expander

VERBOSE = False
MAX_CONCURRENCY = 4
ENABLE_GRAPH = True

lbt.set_logger("dvs")

console = Console()

data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
documents_dir = data_root.joinpath("demo_documents")
duckdb_path = data_root.joinpath("example_graph_dvs.duckdb")
graph_stem_name = "example_graph_dvs"
label_embeddings_path = data_root.joinpath("label_embeddings.json")

if duckdb_path.is_file():
    duckdb_path.unlink()

dvs_settings = dvs.Settings(DUCKDB_PATH=str(duckdb_path))

dvs_client = dvs.DVS(
    dvs_settings,
    model=oai_emb_model.OpenAIEmbeddingsModel(
        model=oai_emb_model.EmbeddingModelType.TEXT_EMBEDDING_3_SMALL,
        openai_client=openai.OpenAI(),
        cache=diskcache.Cache("cache/dvs/embeddings.cache"),
    ),
    model_settings=oai_emb_model.ModelSettings(dimensions=512),
    enable_graph=ENABLE_GRAPH,
    verbose=VERBOSE,
)

openai_client = openai.AsyncOpenAI()
chat_model = agents.OpenAIResponsesModel(
    model="gpt-4.1-nano",
    openai_client=openai_client,
)
emb_model = AsyncOpenAIEmbeddingsModel("text-embedding-3-small", openai_client)
emb_settings = EmbeddingModelSettings(dimensions=512)

aps_agent = APSAgent()
ner_agent = NerAgent()


def load_documents(directory_path: pathlib.Path | str) -> list[Document]:
    documents: list[Document] = load_documents_from_directory(directory_path)
    if not documents:
        raise ValueError(f"No documents found in {directory_path}")
    return documents


async def main():
    console.rule("[bold green]Graph DVS Construction Pipeline[/bold green]")

    # --- Step 1: Document Loading ---
    console.rule("[bold blue]Step 1: Document Loading[/bold blue]")
    raw_documents = load_documents(documents_dir)

    # --- Step 2: Build DVS ---
    console.rule("[bold blue]Step 2: Build DVS[/bold blue]")
    created_result = dvs_client.add(raw_documents)
    console.log(f"Created DVS result: {created_result}")

    # --- Step 2.5: Optional Draw embeddings on 2D plane ---
    console.rule("[bold blue]Step 2.5: Draw embeddings on 2D plane[/bold blue]")
    points = [p for p in dvs_client.db.points.gen(limit=100, with_embedding=True)]
    label_embeddings: typing.List[typing.Tuple[str, typing.List[float]]] = []
    for p in points:
        doc = dvs_client.db.documents.retrieve(p.document_id)
        label_embeddings.append((f"{doc.name}:{doc.chunk_index}", p.to_python()))
    label_embeddings_path.write_text(json.dumps(label_embeddings))

    # --- Step 3: Build Graph ---
    console.rule("[bold blue]Step 3: Build Graph[/bold blue]")
    graph = await build_graph_from_documents(
        [doc for doc in dvs_client.db.documents.gen(verbose=VERBOSE)],
        chat_model=chat_model,
        embeddings_model=emb_model,
        embeddings_model_settings=emb_settings,
        aps_agent=aps_agent,
        ner_agent=ner_agent,
        max_concurrency=MAX_CONCURRENCY,
    )
    dump_graph(graph, data_root.joinpath(f"{graph_stem_name}.json"), format="node_link")
    graph_path = dump_graph(
        graph, data_root.joinpath(f"{graph_stem_name}.gexf"), format="gexf"
    )
    graph = load_graph(graph_path, format="gexf")  # noqa
    console.log(f"Built Graph: {graph_path}")

    # --- Step 3.5: Read Graph from file ---
    console.rule("[bold blue]Step 3.5: Read Graph from file[/bold blue]")
    graph_path = data_root.joinpath(f"{graph_stem_name}.gexf")
    graph = load_graph(graph_path, format="gexf")
    console.log(f"Read Graph: {graph_path}")
    nodes: list[Node] = [
        Node.model_validate(node_data) for _, node_data in graph.nodes(data=True)
    ]
    edges: list[Edge] = [
        Edge.model_validate(edge_data) for _, _, edge_data in graph.edges(data=True)
    ]
    console.log(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

    # --- Step 4: Add Nodes and Edges to DVS ---
    console.rule("[bold blue]Step 4: Add Nodes and Edges to DVS[/bold blue]")
    dvs_client.db.graph.nodes.bulk_create(nodes)
    dvs_client.db.graph.edges.bulk_create(edges)

    # --- Step 5: Query Graph ---
    console.rule("[bold blue]Step 5: Query Graph[/bold blue]")
    head_edges = dvs_client.db.graph.edges.list(relation="related_to", limit=5).data
    peek_edge = head_edges[0]
    print(f"Peek Edge: {peek_edge}")

    results = dvs_client.db.graph.get_neighbors(peek_edge.from_node_id, limit=3)
    print(f"There are {len(results)} neighbors by {peek_edge.from_node_id=}")

    results = dvs_client.db.graph.get_shortest_paths(peek_edge.from_node_id, limit=3)
    print(f"There are {len(results)} shortest paths by {peek_edge.from_node_id=}")
    for result in results:
        print(f"Distance: {result[-1]}, From: {result[0].label}, To: {result[1].label}")

    results = dvs_client.db.graph.local_clustering_coefficient(
        relation="related_to", limit=3
    )
    print(f"There are {len(results)} local clustering coefficients")
    for result in results:
        print(
            f"Local Clustering Coefficient: {result[1]}, "
            + f"Node: {dvs_client.db.graph.nodes.retrieve(result[0],verbose=False)}"
        )

    results = dvs_client.db.graph.weakly_connected_component(
        relation="related_to", limit=3
    )
    print(f"There are {len(results)} weakly connected components")
    for result in results:
        print(
            f"Weakly Connected Component: {result[1]}, "
            + f"Node: {dvs_client.db.graph.nodes.retrieve(result[0],verbose=VERBOSE)}"
        )

    results = dvs_client.db.graph.pagerank(relation="related_to", limit=3)
    print(f"There are {len(results)} pageranks")
    for result in results:
        print(
            f"Pagerank: {result[1]}, "
            + f"Node: {dvs_client.db.graph.nodes.retrieve(result[0],verbose=VERBOSE)}"
        )

    # --- Step 6: Graph-RAG Search Demo ---
    console.rule("[bold blue]Step 6: Graph-RAG Search Demo[/bold blue]")

    # Compare traditional RAG vs Graph-RAG search results
    query: str = "Where are Michelin restaurants?"

    console.rule("[bold yellow]Traditional RAG Search[/bold yellow]")
    traditional_results = await dvs_client.search(query, top_k=3, verbose=VERBOSE)
    for i, (point, doc, score) in enumerate(traditional_results, 1):
        print(f"{i}. Score: {score:.3f}, Document: {doc.name}")
        print(f"   Content: {doc.content[:100]}...")

    console.rule("[bold green]Graph-RAG Search (Strategy 1)[/bold green]")
    graph_rag_results = await dvs_client.graph_rag_search_vector_expansion(
        query,
        top_k=3,
        graph_expansion_depth=1,
        vector_weight=0.7,
        graph_weight=0.3,
        verbose=VERBOSE,
    )

    for i, item in enumerate(graph_rag_results, 1):
        print(f"{i}. Score: {item.score:.3f}, Document: {item.document.name}")
        print(f"   Content: {item.document.content[:100]}...")

    console.rule("[bold cyan]Comparison Summary[/bold cyan]")
    print("Graph-RAG can find:")
    print("• Documents connected through entity relationships")
    print("• Semantically related content via graph traversal")
    print("• Better context by considering document interconnections")

    # --- Step 7: Graph-RAG Strategy 2 Demo ---
    console.rule("[bold magenta]Step 7: Graph-RAG Strategy 2 Demo[/bold magenta]")
    console.rule("[bold yellow]Strategy 2: Graph-Guided Vector Search[/bold yellow]")

    print("🎯 Strategy 2 Flow:")
    print("   Query → PageRank → Important Nodes → Vector Search → Combined Scoring")
    print("")

    # Test different centrality thresholds
    thresholds = [0.3, 0.5, 0.7]
    strategy2_results = {}

    for threshold in thresholds:
        console.rule(
            f"[bold blue]Testing Centrality Threshold: {threshold}[/bold blue]"
        )

        try:
            results = await dvs_client.graph_rag_search_graph_guided(
                query=query, top_k=3, centrality_threshold=threshold, verbose=VERBOSE
            )

            strategy2_results[threshold] = results

            print(f"\n📊 Results for threshold {threshold}:")
            for i, item in enumerate(results, 1):
                print(f"{i}. Score: {item.score:.3f}, Document: {item.document.name}")
                print(f"   Content: {item.document.content[:100]}...")

        except Exception as e:
            print(f"⚠️ Error with threshold {threshold}: {e}")
            strategy2_results[threshold] = []

        print("")

    # --- Step 8: Strategy Comparison ---
    console.rule("[bold green]Step 8: Strategy Comparison[/bold green]")

    print("🔍 Comparison of all strategies:")
    print("")

    # Traditional search results
    print("1️⃣ Traditional Vector Search:")
    for i, (point, doc, score) in enumerate(traditional_results, 1):
        print(f"   {i}. {doc.name}: {score:.3f}")
    print("")

    # Strategy 1 results
    print("2️⃣ Graph-RAG Strategy 1 (Vector + Graph Expansion):")
    for i, item in enumerate(graph_rag_results, 1):
        print(f"   {i}. {item.document.name}: {item.score:.3f}")
    print("")

    # Strategy 2 results for different thresholds
    for threshold, results in strategy2_results.items():
        if results:
            print(f"3️⃣ Graph-RAG Strategy 2 (Threshold {threshold}):")
            for i, item in enumerate(results, 1):
                print(f"   {i}. {item.document.name}: {item.score:.3f}")
            print("")

    # --- Step 9: Strategy 2 Deep Analysis ---
    console.rule("[bold purple]Step 9: Strategy 2 Deep Analysis[/bold purple]")

    # Analyze PageRank results
    try:
        print("📊 Analyzing PageRank centrality in the graph...")
        pagerank_results = dvs_client.db.graph.pagerank(
            relation="related_to", limit=10, verbose=False
        )

        print("\n🏆 Top 10 Most Important Entities (by PageRank):")
        for i, (node_id, score) in enumerate(pagerank_results[:10], 1):
            try:
                node = dvs_client.db.graph.nodes.retrieve(node_id, verbose=False)
                print(f"   {i}. {node.label}: {score:.4f}")
            except Exception:
                print(f"   {i}. {node_id}: {score:.4f}")

        print(f"\n📈 Total entities analyzed: {len(pagerank_results)}")

    except Exception as e:
        print(f"⚠️ Error analyzing PageRank: {e}")

    # --- Step 10: Performance Insights ---
    console.rule("[bold cyan]Step 10: Performance Insights[/bold cyan]")

    insights = [
        "🎯 Strategy 2 excels at finding authoritative, well-connected content",
        "📊 PageRank helps discover content through 'important' entities",
        "🔗 Graph-guided search finds semantically related but not obvious content",
        "⚖️ Combined scoring balances vector similarity with graph importance",
        "🎚️ Threshold tuning affects precision vs. recall trade-off",
        "🚀 Strategy 2 is great for expert-level or authoritative content discovery",
    ]

    for insight in insights:
        print(f"💡 {insight}")

    print("")
    print("✨ Demo completed! Strategy 2 showcases how graphs can intelligently")
    print("   guide vector search to find more relevant and authoritative content.")

    # --- Step 11: Graph-RAG Strategy 3 Demo ---
    console.rule("[bold magenta]Step 11: Graph-RAG Strategy 3 Demo[/bold magenta]")
    console.rule("[bold yellow]Strategy 3: Hybrid Scoring[/bold yellow]")

    print("🎯 Strategy 3 combines:")
    print("   Vector similarity + Graph importance (PageRank) + Graph distance")

    try:
        hybrid_results = await dvs_client.graph_rag_search_hybrid_scoring(
            query=query,
            top_k=3,
            vector_weight=0.7,
            graph_importance_weight=0.2,
            graph_distance_weight=0.1,
            verbose=VERBOSE,
        )

        print("\n📊 Results for Strategy 3 (Hybrid Scoring):")
        for i, item in enumerate(hybrid_results, 1):
            print(f"   {i}. {item.document.name}: {item.score:.3f}")
            print(f"      Content: {item.document.content[:100]}...")
    except Exception as e:
        print(f"⚠️ Error running Strategy 3: {e}")

    # --- Step 12: Strategy 4 Demo (Iterative Refinement with LLM Expander) ---
    console.rule("[bold magenta]Step 12: Strategy 4 Demo[/bold magenta]")
    console.rule("[bold yellow]Strategy 4: Iterative Refinement (LLM expansion)")

    try:
        expander = build_llm_query_expander(
            chat_model,
            max_suggestions=3,
            system_prompt=None,
            verbose=VERBOSE,
        )
        iter4_results: list[GraphRAGResult] = (
            await dvs_client.graph_rag_search_iterative_refinement(
                query=query,
                top_k=3,
                max_iterations=3,
                refinement_threshold=0.05,
                expansions_per_iter=3,
                query_expander=expander,
                verbose=VERBOSE,
            )
        )

        print("\n📊 Results for Strategy 4 (Iterative Refinement):")
        for i, item in enumerate(iter4_results, 1):
            name = item.document.name
            score_str = f"{item.score:.3f}"
            iters_str = str(item.iterations)
            print(f"   {i}. {name}: {score_str} (iters={iters_str})")
            print(f"      Content: {item.document.content[:100]}...")
    except Exception as e:
        print(f"⚠️ Error running Strategy 4: {e}")

    # --- Step 13: Strategy 5 Demo (Context-Aware Expansion) ---
    console.rule("[bold magenta]Step 13: Graph-RAG Strategy 5 Demo[/bold magenta]")
    console.rule("[bold yellow]Strategy 5: Context-Aware Expansion")

    print("🎯 Strategy 5 uses: baseline seeds → is_from expand → context centroid")
    ctx_threshold: float = 0.0

    try:
        ctx_results: list[GraphRAGResult] = (
            await dvs_client.graph_rag_search_context_aware(
                query=query,
                top_k=3,
                context_similarity_threshold=ctx_threshold,
                max_expansion_steps=1,
                verbose=True,
            )
        )

        print("\n📊 Results for Strategy 5 (Context-Aware):")
        for i, item in enumerate(ctx_results, 1):
            name = item.document.name
            score_str = f"{item.score:.3f}"
            v_str = f"{(item.vector_score or 0.0):.3f}"
            g_str = f"{(item.graph_score or 0.0):.3f}"
            print(f"   {i}. {name}: {score_str} (vector={v_str}, context={g_str})")
            print(f"      Content: {item.document.content[:100]}...")
    except Exception as e:
        print(f"⚠️ Error running Strategy 5: {e}")


if __name__ == "__main__":
    asyncio.run(main())
