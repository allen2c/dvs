import asyncio
import hashlib
import logging
import pathlib
import typing

import agents
import cachetic
import pydantic
from rich.pretty import pretty_repr

if typing.TYPE_CHECKING:
    from aps_agent import APSAgent, APSResult

    from dvs.types.document import Document
    from dvs.types.fact import Fact


logger = logging.getLogger(__name__)


async def get_facts(
    documents: typing.List["Document"],
    *,
    aps_agent: typing.Optional["APSAgent"] = None,
    model: agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel,
    max_concurrency: int = 1,
    cache: cachetic.Cachetic["APSResult"] | None = None,
    verbose: bool = False,
) -> typing.List["Fact"]:
    """Extract facts from documents using APS agent."""
    from aps_agent import APSAgent, APSResult

    from dvs.types.fact import Fact
    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    aps_agent = aps_agent or APSAgent()
    if cache is None:
        cache = cachetic.Cachetic(
            object_type=pydantic.TypeAdapter(APSResult),
            cache_url=pathlib.Path(".cache/aps.cache"),
        )

    all_facts: typing.List[Fact] = []

    async def run_aps_agent(doc: "Document"):
        """Run APS agent on a single document to extract facts."""
        _input_text = doc.content
        logger.info(f"Running APS on {pretty_repr(doc.name, max_string=32)}...")

        cache_key = (
            "aps:"
            + f"{model.model}:"
            + hashlib.sha256(_input_text.encode()).hexdigest()
        )
        might_result = await asyncio.to_thread(cache.get, cache_key)
        if might_result:
            result = might_result
        else:
            result = await aps_agent.run(_input_text, model=model, verbose=verbose)
            await asyncio.to_thread(cache.set, cache_key, result)

        facts: list[Fact] = []
        for fact in result.facts:
            facts.append(Fact(fact=fact.fact, document_id=doc.document_id))
        return facts

    aps_result_tasks = [run_aps_agent(doc) for doc in documents]

    facts_results = await gather_with_concurrency_limit(
        aps_result_tasks, limit=max_concurrency
    )
    all_facts.extend(fact for fact_list in facts_results for fact in fact_list)

    logger.info(f"✅ Extracted a total of {len(all_facts)} facts.")

    return all_facts
