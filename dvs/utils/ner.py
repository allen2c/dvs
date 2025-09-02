import asyncio
import logging
import typing

import agents

if typing.TYPE_CHECKING:
    from ner_agent import NerAgent

    from dvs.types.entity import Entity
    from dvs.types.fact import Fact
    from dvs.types.triplet import Triplet


logger = logging.getLogger(__name__)


async def extract_relations(
    facts: typing.List["Fact"],
    *,
    ner_agent: typing.Optional["NerAgent"] = None,
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> list["Triplet"]:

    from dvs.types.triplet import Triplet
    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    ner_agent = ner_agent or NerAgent()

    total_count = len(facts)
    complete_count = 0

    counter_lock = asyncio.Lock()

    async def run_extract_relations(fact: Fact) -> list["Triplet"]:
        nonlocal complete_count
        async with counter_lock:
            current_idx = complete_count = complete_count + 1
        logger.debug(f"Extracting relations for fact {current_idx}/{total_count}")
        result = await ner_agent.extract_relations(
            fact.fact, model=model, verbose=verbose
        )
        return [
            Triplet(
                subject=triplet.subject,
                relation=triplet.relation,
                object=triplet.object,
                document_id=fact.document_id,
            )
            for triplet in result.triplets
        ]

    relation_tasks = [run_extract_relations(fact) for fact in facts]

    triplets_results: list[list["Triplet"]] = await gather_with_concurrency_limit(
        relation_tasks, limit=max_concurrency
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
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> list["Entity"]:
    from dvs.types.entity import Entity
    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    ner_agent = ner_agent or NerAgent()

    total_count = len(facts)
    complete_count = 0
    counter_lock = asyncio.Lock()

    async def run_extract_entities(fact: Fact) -> list["Entity"]:
        nonlocal complete_count
        async with counter_lock:
            current_idx = complete_count = complete_count + 1
        logger.debug(f"Extracting entities for fact {current_idx}/{total_count}")
        result = await ner_agent.run(fact.fact, model=model, verbose=verbose)
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

    entities_results: list[list[Entity]] = await gather_with_concurrency_limit(
        entity_tasks, limit=max_concurrency
    )

    all_entities: list[Entity] = [
        ent for entities in entities_results for ent in entities
    ]
    logger.info(f"✅ Extracted {len(all_entities)} raw entities.")

    return all_entities
