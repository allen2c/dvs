import functools
import time


class Timer:
    def __init__(self):
        """Initialize the timer with default values."""
        self.time_start: float | None = None
        self.time_end: float | None = None
        self._duration: float | None = None

    def __enter__(self):
        """Start the timer when entering context."""
        self.time_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer when exiting context."""
        self.time_end = time.perf_counter()

    @functools.cached_property
    def duration(self) -> float:
        """Get the elapsed time in seconds."""
        if self.time_start is None or self.time_end is None:
            raise ValueError("Timer not started or stopped")
        return self.time_end - self.time_start
