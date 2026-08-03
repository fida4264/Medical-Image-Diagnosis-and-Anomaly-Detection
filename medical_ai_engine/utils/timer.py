"""
Medical AI Engine — High-Precision Monotonic Timer

Provides context manager and class for measuring nanosecond/millisecond execution time.
"""

from __future__ import annotations

import time
from typing import Optional


class EngineTimer:
    """
    High-precision timer for measuring code block latency.
    Can be used as a context manager or explicit start/stop object.

    Usage:
        with EngineTimer() as timer:
            # work
        print(timer.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start_ns: Optional[int] = None
        self._end_ns: Optional[int] = None

    def start(self) -> EngineTimer:
        """Start or restart the timer."""
        self._start_ns = time.perf_counter_ns()
        self._end_ns = None
        return self

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in milliseconds.
        """
        if self._start_ns is None:
            return 0.0
        self._end_ns = time.perf_counter_ns()
        return self.elapsed_ms

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        if self._start_ns is None:
            return 0
        end = self._end_ns if self._end_ns is not None else time.perf_counter_ns()
        return end - self._start_ns

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return round(self.elapsed_ns / 1_000_000.0, 3)

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return round(self.elapsed_ns / 1_000_000_000.0, 6)

    def __enter__(self) -> EngineTimer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
