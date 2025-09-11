import asyncio
import hashlib
import json
import logging
import pathlib
import typing

import agents
import cachetic
import openai_embeddings_model as oai_emb_model
import pydantic

if typing.TYPE_CHECKING:
    from ner_agent import NerAgent, SynonymsAndCanonicalNameResult


logger = logging.getLogger(__name__)


async def get_valid_canonical_map(
    labels: list[str],
    *,
    embedding_model: (
        oai_emb_model.OpenAIEmbeddingsModel | oai_emb_model.AsyncOpenAIEmbeddingsModel
    ),
    model_settings: oai_emb_model.ModelSettings,
    clustering_eps: float = 0.3,
    clustering_min_samples: int = 2,
    ner_agent: typing.Optional["NerAgent"] = None,
    chat_model: agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel,
    model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    cache: cachetic.Cachetic["SynonymsAndCanonicalNameResult"] | None = None,
    verbose: bool = False,
) -> typing.Dict[str, str]:
    import sklearn.cluster
    from ner_agent import NerAgent, SynonymsAndCanonicalNameResult

    from dvs.utils.format_string import format_string
    from dvs.utils.gather_with_concurrency_limit import gather_with_semaphore

    ner_agent = ner_agent or NerAgent()
    if cache is None:
        cache = cachetic.Cachetic(
            object_type=pydantic.TypeAdapter(SynonymsAndCanonicalNameResult),
            cache_url=pathlib.Path(".cache/synonyms_and_canonical_name.cache"),
        )

    # 1. Embedding
    if isinstance(embedding_model, oai_emb_model.OpenAIEmbeddingsModel):
        response = await asyncio.to_thread(
            embedding_model.get_embeddings, input=labels, model_settings=model_settings
        )
    else:
        response = await embedding_model.get_embeddings(
            input=labels, model_settings=model_settings
        )
    embeddings = response.to_numpy()

    # 2. Clustering
    clustering = sklearn.cluster.DBSCAN(
        eps=clustering_eps, min_samples=clustering_min_samples, metric="cosine"
    ).fit(embeddings)
    clusters_labels = clustering.labels_

    clusters: typing.Dict[int, list[str]] = {}
    for i, label in enumerate(clusters_labels):
        label_int = int(label)  # Convert numpy int to Python int
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append(labels[i])
    logger.info(f"✅ Clustered entities into {len(set(clusters_labels)) - 1} groups.")

    # 3. LLM Validation and Canonical Map Generation
    canonical_map: typing.Dict[str, str] = {}
    validation_tasks = {
        label: run_analyze_synonyms_and_canonical_name_with_cache(
            items,
            ner_agent=ner_agent,
            chat_model=chat_model,
            cache=cache,
            verbose=verbose,
        )
        for label, items in clusters.items()
        if label != -1 and len(items) > 1
    }
    validation_results = await gather_with_semaphore(
        validation_tasks.values(), semaphore=model_semaphore
    )
    validated_clusters = dict(zip(validation_tasks.keys(), validation_results))

    for label, result in validated_clusters.items():
        items = clusters[label]
        if result.is_synonymous and result.canonical_name:
            for item in items:
                canonical_map[item] = result.canonical_name
                canonical_map[format_string(item)] = result.canonical_name

    # Handle non-synonyms, single-item clusters, and noise (they map to themselves)
    for _label in labels:
        if _label not in canonical_map:
            canonical_map[_label] = _label
    logger.info(f"✅ Generated canonical map for all {len(labels)} labels.")

    return canonical_map


async def run_analyze_synonyms_and_canonical_name_with_cache(
    items: list[str],
    *,
    ner_agent: "NerAgent",
    chat_model: agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel,
    cache: cachetic.Cachetic["SynonymsAndCanonicalNameResult"],
    verbose: bool = False,
) -> "SynonymsAndCanonicalNameResult":
    cache_key = (
        "analyze_synonyms_and_canonical_name:"
        + f"{chat_model.model}:"
        + hashlib.sha256(json.dumps(items).encode()).hexdigest()
    )
    result = await asyncio.to_thread(cache.get, cache_key)
    if result:
        return result

    result = await ner_agent.analyze_synonyms_and_canonical_name(
        items, model=chat_model, verbose=verbose
    )
    await asyncio.to_thread(cache.set, cache_key, result)
    return result
