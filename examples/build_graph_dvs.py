import asyncio
import pathlib

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
from dvs.types.graphrag_result import GraphRAGResult
from dvs.utils.load_documents_from_directory import load_documents_from_directory
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

    # --- Step 3: Build Graph ---
    console.rule("[bold blue]Step 3: Build Graph[/bold blue]")
    await dvs_client.db.graph.rebuild_graph(
        chat_model=chat_model,
        document_semaphore=asyncio.Semaphore(MAX_CONCURRENCY),
        model_semaphore=asyncio.Semaphore(MAX_CONCURRENCY),
        verbose=VERBOSE,
    )
    dvs_client.db.graph.export(
        data_root.joinpath(f"{graph_stem_name}.json"), format="node_link"
    )
    dvs_client.db.graph.export(
        data_root.joinpath(f"{graph_stem_name}.gexf"), format="gexf"
    )

    # --- Step 4: Unified Graph-RAG Comparison (Strategies 1-5) ---
    console.rule(
        "[bold cyan]Unified Graph-RAG Comparison (Strategies 1-5) - Quick Mode[/bold cyan]"  # noqa: E501
    )

    query: str = "Where are Michelin restaurants?"

    # Parameters for fairness
    topk: int = 3
    graph_depth: int = 1
    vec_w: float = 0.7
    graph_w: float = 0.3
    centrality_th: float = 0.5
    hybrid_vec_w: float = 0.7
    hybrid_imp_w: float = 0.2
    hybrid_dist_w: float = 0.1
    ctx_th: float = 0.5

    def fmt_item(i: int, item: GraphRAGResult) -> str:
        s = f"{i}. {item.document.name}: {item.score:.3f}"
        if item.vector_score is not None:
            s += f" | v={item.vector_score:.3f}"
        if item.graph_score is not None:
            s += f" | g={item.graph_score:.3f}"
        if item.iterations is not None:
            s += f" | iters={item.iterations}"
        return s

    try:
        s1 = await dvs_client.db.graph.search_vector_expansion(
            query,
            top_k=topk,
            graph_expansion_depth=graph_depth,
            vector_weight=vec_w,
            graph_weight=graph_w,
            verbose=VERBOSE,
        )

        s2 = await dvs_client.db.graph.search_graph_guided(
            query,
            top_k=topk,
            centrality_threshold=centrality_th,
            vector_weight=vec_w,
            graph_weight=graph_w,
            verbose=VERBOSE,
        )

        s3 = await dvs_client.db.graph.search_hybrid_scoring(
            query,
            top_k=topk,
            vector_weight=hybrid_vec_w,
            graph_importance_weight=hybrid_imp_w,
            graph_distance_weight=hybrid_dist_w,
            verbose=VERBOSE,
        )

        expander = build_llm_query_expander(
            chat_model,
            max_suggestions=3,
            system_prompt=None,
            verbose=VERBOSE,
        )

        s4 = await dvs_client.db.graph.search_iterative_refinement(
            query,
            top_k=topk,
            max_iterations=1,
            refinement_threshold=0.01,
            expansions_per_iter=1,
            query_expander=expander,
            verbose=VERBOSE,
        )

        s5 = await dvs_client.db.graph.search_context_aware(
            query,
            top_k=topk,
            context_similarity_threshold=ctx_th,
            max_expansion_steps=1,
            verbose=VERBOSE,
        )

        print("\n📊 Unified Comparison (top-3):")
        print("1) Strategy 1 - Vector + Graph Expansion")
        for i, item in enumerate(s1, 1):
            print("   " + fmt_item(i, item))
        print("2) Strategy 2 - Graph-Guided Vector Search")
        for i, item in enumerate(s2, 1):
            print("   " + fmt_item(i, item))
        print("3) Strategy 3 - Hybrid Scoring")
        for i, item in enumerate(s3, 1):
            print("   " + fmt_item(i, item))
        print("4) Strategy 4 - Iterative Refinement")
        for i, item in enumerate(s4, 1):
            print("   " + fmt_item(i, item))
        print("5) Strategy 5 - Context-Aware Expansion")
        for i, item in enumerate(s5, 1):
            print("   " + fmt_item(i, item))

    except Exception as e:
        print(f"⚠️ Error in unified comparison: {e}")

    return


if __name__ == "__main__":
    asyncio.run(main())
