import logging
import typing

import agents

if typing.TYPE_CHECKING:
    from aps_agent import Fact
    from ner_agent import Entity, NerAgent, Triplet


logger = logging.getLogger(__name__)


async def extract_relations(
    facts: typing.Union[
        typing.List["Fact"],
        typing.List[typing.Text],
        typing.List[typing.Union["Fact", typing.Text]],
    ],
    *,
    ner_agent: typing.Optional["NerAgent"] = None,
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> list["Triplet"]:

    from ner_agent import Triplet

    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    all_facts = [fact if isinstance(fact, typing.Text) else fact.fact for fact in facts]
    ner_agent = ner_agent or NerAgent()
    complete_count = 0

    async def run_extract_relations(fact: str):
        nonlocal complete_count
        current_idx = complete_count = complete_count + 1
        logger.debug(f"Extracting relations for fact {current_idx}/{len(all_facts)}")
        return await ner_agent.extract_relations(fact, model=model, verbose=verbose)

    relation_tasks = [run_extract_relations(fact) for fact in all_facts]
    results = await gather_with_concurrency_limit(relation_tasks, limit=max_concurrency)

    all_triplets: list[Triplet] = []
    for res in results:
        all_triplets.extend(res.triplets)
    logger.info(f"✅ Extracted {len(all_triplets)} raw triplets.")

    return all_triplets


async def extract_entities(
    facts: typing.Union[
        typing.List["Fact"],
        typing.List[typing.Text],
        typing.List[typing.Union["Fact", typing.Text]],
    ],
    *,
    ner_agent: typing.Optional["NerAgent"] = None,
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> list["Entity"]:
    from ner_agent import Entity

    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    all_facts = [fact if isinstance(fact, typing.Text) else fact.fact for fact in facts]
    ner_agent = ner_agent or NerAgent()
    complete_count = 0

    async def run_extract_entities(fact: str):
        nonlocal complete_count
        current_idx = complete_count = complete_count + 1
        logger.debug(f"Extracting entities for fact {current_idx}/{len(all_facts)}")
        return await ner_agent.run(fact, model=model, verbose=verbose)

    entity_tasks = [run_extract_entities(fact) for fact in all_facts]
    results = await gather_with_concurrency_limit(entity_tasks, limit=max_concurrency)

    all_entities: list[Entity] = [ent for result in results for ent in result.entities]
    logger.info(f"✅ Extracted {len(all_entities)} raw entities.")

    return all_entities
