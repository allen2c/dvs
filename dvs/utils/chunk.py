import itertools
from typing import Generator, Iterable, Tuple, TypeVar

T = TypeVar("T")


def chunks(
    items: Iterable[T], batch_size: int = 100
) -> Generator[Tuple[T, ...], None, None]:
    """A helper function to break an iterable into chunks of size batch_size."""

    it = iter(items)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
