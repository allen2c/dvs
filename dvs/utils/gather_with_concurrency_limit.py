import asyncio
import typing

T = typing.TypeVar("T")


async def gather_with_concurrency_limit(
    awaitables: typing.Iterable[typing.Awaitable[T]], limit: int
) -> list[T]:
    semaphore: asyncio.Semaphore = asyncio.Semaphore(limit)

    async def run_with_semaphore(coro: typing.Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    wrapped_tasks: list[typing.Awaitable[T]] = [
        run_with_semaphore(coro) for coro in awaitables
    ]
    return await asyncio.gather(*wrapped_tasks)
