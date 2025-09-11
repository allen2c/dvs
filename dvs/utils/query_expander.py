import asyncio
import hashlib
import pathlib
import typing

import agents
import cachetic
import jinja2
import pydantic


class QueryExpansionResult(pydantic.BaseModel):
    """Expanded queries from LLM in structured form."""

    suggestions: list[str]


def build_llm_query_expander(
    model: agents.OpenAIResponsesModel | agents.OpenAIChatCompletionsModel,
    *,
    max_suggestions: int = 3,
    system_prompt: str | None = None,
    cache: cachetic.Cachetic[QueryExpansionResult] | None = None,
    model_settings: agents.ModelSettings | None = None,
    tracing_disabled: bool = True,
    verbose: bool = False,
) -> typing.Callable[[str], "typing.Awaitable[list[str]]"]:
    """Return an async function that expands queries using agents.Runner."""

    if cache is None:
        cache = cachetic.Cachetic(
            object_type=pydantic.TypeAdapter(QueryExpansionResult),
            cache_url=pathlib.Path(".cache/query_expander.cache"),
        )

    sys_prompt = system_prompt or (
        "You are a query expansion helper. Given a user query, return up to N "
        + "short, diverse rephrasings as JSON under 'suggestions'."
    )

    async def expand(query: str) -> list[str]:
        cache_key = (
            "qe:" + f"{model.model}:" + hashlib.sha256(query.encode()).hexdigest()
        )
        might = await asyncio.to_thread(cache.get, cache_key)
        if might:
            return [s for s in might.suggestions][:max_suggestions]

        instructions = jinja2.Template(
            (
                "{{ system_prompt }}\n"
                + "Return JSON strictly matching: {\n"
                + '  "suggestions": string[]\n'
                + "}. Do not include extra text.\n"
                + "N = {{ max_suggestions }}.\n"
                + "User query: {{ query | tojson }}\n"
            )
        ).render(
            system_prompt=sys_prompt,
            max_suggestions=max_suggestions,
            query=query,
        )

        agent = agents.Agent(
            name="query-expander",
            model=model,
            model_settings=model_settings or agents.ModelSettings(),
            instructions=instructions,
            output_type=QueryExpansionResult,
        )

        runner_result = await agents.Runner.run(
            agent,
            instructions,
            run_config=agents.RunConfig(tracing_disabled=tracing_disabled),
        )

        final: QueryExpansionResult = runner_result.final_output_as(
            QueryExpansionResult
        )
        result = QueryExpansionResult(
            suggestions=[s for s in final.suggestions][:max_suggestions]
        )
        await asyncio.to_thread(cache.set, cache_key, result)
        return result.suggestions

    return expand
