import logging
import typing

import agents
from rich.pretty import pretty_repr

if typing.TYPE_CHECKING:
    from aps_agent import APSAgent

    from dvs.types.document import Document
    from dvs.types.fact import Fact


logger = logging.getLogger(__name__)


async def get_facts(
    documents: typing.List["Document"],
    *,
    aps_agent: typing.Optional["APSAgent"] = None,
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> typing.List["Fact"]:
    """Extract facts from documents using APS agent."""
    from aps_agent import APSAgent

    from dvs.types.fact import Fact
    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    aps_agent = aps_agent or APSAgent()
    all_facts: typing.List[Fact] = []

    async def run_aps_agent(doc: "Document"):
        """Run APS agent on a single document to extract facts."""
        _input_text = doc.content
        logger.info(f"Running APS on {pretty_repr(doc.name, max_string=32)}...")

        facts_result = await aps_agent.run(_input_text, model=model, verbose=verbose)
        facts: list[Fact] = []
        for fact in facts_result.facts:
            facts.append(Fact(fact=fact.fact, document_id=doc.document_id))
        return facts

    aps_result_tasks = [run_aps_agent(doc) for doc in documents]

    facts_results = await gather_with_concurrency_limit(
        aps_result_tasks, limit=max_concurrency
    )
    all_facts.extend(fact for fact_list in facts_results for fact in fact_list)

    logger.info(f"✅ Extracted a total of {len(all_facts)} facts.")

    return all_facts
