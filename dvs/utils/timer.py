import functools
import time


class Timer:
    def __init__(self):
        self.time_start: float | None = None
        self.time_end: float | None = None
        self._duration: float | None = None

    def __enter__(self):
        self.time_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.time_end = time.perf_counter()

    @functools.cached_property
    def duration(self) -> float:
        if self.time_start is None or self.time_end is None:
            raise ValueError("Timer not started or stopped")
        return self.time_end - self.time_start
