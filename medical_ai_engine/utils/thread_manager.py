"""
Medical AI Engine — Thread Manager

Manages thread pool execution and worker threads for concurrent engine requests.
Thread-safe.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
import threading
from typing import Callable, Any, Optional


class EngineThreadManager:
    """
    Manages background thread execution and thread pools for MedicalAIEngine.
    """

    def __init__(self, max_workers: int = 16, thread_prefix: str = "MedAI-EngineWorker") -> None:
        self.max_workers = max_workers
        self.thread_prefix = thread_prefix
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Initialize the thread pool executor."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self.thread_prefix,
                )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """
        Submit a callable task to the thread pool.
        """
        with self._lock:
            if self._executor is None:
                self.start()
            assert self._executor is not None
            return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
                self._executor = None

    @property
    def is_running(self) -> bool:
        """Returns True if the executor is active."""
        with self._lock:
            return self._executor is not None
