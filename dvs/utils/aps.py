import logging
import typing

import agents
from rich.pretty import pretty_repr

if typing.TYPE_CHECKING:
    from aps_agent import APSAgent

    from dvs.types.document import Document


logger = logging.getLogger(__name__)


async def get_facts(
    documents: typing.Union[
        typing.List["Document"],
        typing.List[typing.Text],
        typing.List[typing.Union["Document", typing.Text]],
    ],
    *,
    aps_agent: typing.Optional["APSAgent"] = None,
    model: (
        agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | None
    ) = None,
    max_concurrency: int = 1,
    verbose: bool = False,
):
    from aps_agent import APSAgent, Fact

    from dvs.utils.gather_with_concurrency_limit import gather_with_concurrency_limit

    aps_agent = aps_agent or APSAgent()
    all_facts: typing.List[Fact] = []

    async def run_aps_agent(doc: typing.Union["Document", typing.Text]):
        _input_text: str
        if isinstance(doc, typing.Text):
            _input_text = doc
            logger.info(f"Running APS on {pretty_repr(doc, max_string=32)}...")
        else:
            _input_text = doc.content
            logger.info(f"Running APS on {pretty_repr(doc.name, max_string=32)}...")

        return await aps_agent.run(_input_text, model=model, verbose=verbose)

    aps_result_tasks = [run_aps_agent(doc) for doc in documents]

    aps_results = await gather_with_concurrency_limit(
        aps_result_tasks, limit=max_concurrency
    )
    for aps_result in aps_results:
        all_facts.extend(aps_result.facts)

    logger.info(f"✅ Extracted a total of {len(all_facts)} facts.")

    return all_facts
