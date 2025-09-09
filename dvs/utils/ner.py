import asyncio
import hashlib
import logging
import pathlib
import time
import typing

import agents
import cachetic
import pydantic

if typing.TYPE_CHECKING:
    from ner_agent import NerAgent, NerResult, RelationExtractionResult

    from dvs.types.entity import Entity
    from dvs.types.fact import Fact
    from dvs.types.triplet import Triplet


logger = logging.getLogger(__name__)


async def extract_relations(
    facts: typing.List["Fact"],
    *,
    ner_agent: typing.Optional["NerAgent"] = None,
    model: agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel,
    model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    cache: cachetic.Cachetic["RelationExtractionResult"] | None = None,
    verbose: bool = False,
) -> list["Triplet"]:

    from ner_agent import RelationExtractionResult

    from dvs.types.triplet import Triplet
    from dvs.utils.gather_with_concurrency_limit import gather_with_semaphore

    ner_agent = ner_agent or NerAgent()
    if cache is None:
        cache = cachetic.Cachetic(
            object_type=pydantic.TypeAdapter(RelationExtractionResult),
            cache_url=pathlib.Path(".cache/relation_extraction.cache"),
        )

    total_count = len(facts)
    complete_count = 0
    counter_lock = asyncio.Lock()
    pc = time.perf_counter()

    async def run_extract_relations(fact: "Fact") -> list["Triplet"]:
        nonlocal complete_count
        nonlocal pc
        async with counter_lock:
            current_idx = complete_count = complete_count + 1

        if current_pc := time.perf_counter() - pc > 10:
            logger.debug(f"Extracting relations for fact {current_idx}/{total_count}")
            pc = current_pc

        cache_key = (
            "extract_relations:"
            + f"{model.model}:"
            + hashlib.sha256(fact.fact.encode()).hexdigest()
        )
        might_result = await asyncio.to_thread(cache.get, cache_key)
        if might_result:
            result = might_result
        else:
            result = await ner_agent.extract_relations(
                fact.fact, model=model, verbose=verbose
            )
            await asyncio.to_thread(cache.set, cache_key, result)

        return [
            Triplet(
                subject=triplet.subject,
                relation=(
                    triplet.relation
                    if triplet.relation in ("is_a", "has_a", "related_to", "is_from")
                    else "related_to"
                ),
                object=triplet.object,
                document_id=fact.document_id,
            )
            for triplet in result.triplets
        ]

    relation_tasks = [run_extract_relations(fact) for fact in facts]

    triplets_results: list[list["Triplet"]] = await gather_with_semaphore(
        relation_tasks, semaphore=model_semaphore
    )

    all_triplets: list[Triplet] = []
    for _triplets in triplets_results:
        all_triplets.extend(_triplets)
    logger.info(f"✅ Extracted {len(all_triplets)} raw triplets.")

    return all_triplets


async def extract_entities(
    facts: typing.List["Fact"],
    *,
    ner_agent: typing.Optional["NerAgent"] = None,
    model: agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel,
    model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    cache: cachetic.Cachetic["NerResult"] | None = None,
    verbose: bool = False,
) -> list["Entity"]:
    from ner_agent import NerResult

    from dvs.types.entity import Entity
    from dvs.utils.gather_with_concurrency_limit import gather_with_semaphore

    ner_agent = ner_agent or NerAgent()
    if cache is None:
        cache = cachetic.Cachetic(
            object_type=pydantic.TypeAdapter(NerResult),
            cache_url=pathlib.Path(".cache/ner.cache"),
        )

    total_count = len(facts)
    complete_count = 0
    counter_lock = asyncio.Lock()
    pc = time.perf_counter()

    async def run_extract_entities(fact: "Fact") -> list["Entity"]:
        nonlocal complete_count
        nonlocal pc
        async with counter_lock:
            current_idx = complete_count = complete_count + 1

        if current_pc := time.perf_counter() - pc > 10:
            logger.debug(f"Extracting entities for fact {current_idx}/{total_count}")
            pc = current_pc

        cache_key = (
            "extract_entities:"
            + f"{model.model}:"
            + hashlib.sha256(fact.fact.encode()).hexdigest()
        )
        might_result = await asyncio.to_thread(cache.get, cache_key)
        if might_result:
            result = might_result
        else:
            result = await ner_agent.run(fact.fact, model=model, verbose=verbose)
            await asyncio.to_thread(cache.set, cache_key, result)

        return [
            Entity(
                name=entity.name,
                value=entity.value,
                document_id=fact.document_id,
                start=entity.start,
                end=entity.end,
            )
            for entity in result.entities
        ]

    entity_tasks = [run_extract_entities(fact) for fact in facts]

    entities_results: list[list[Entity]] = await gather_with_semaphore(
        entity_tasks, semaphore=model_semaphore
    )

    all_entities: list[Entity] = [
        ent for entities in entities_results for ent in entities
    ]
    logger.info(f"✅ Extracted {len(all_entities)} raw entities.")

    return all_entities
