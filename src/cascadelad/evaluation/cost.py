"""Wall-clock timers for cost reporting (`infer_ms_per_window` in metrics.json).

Token counting for LLM runs is added in milestone M2.
"""

import time


class Timer:
    """Context manager measuring wall-clock time in milliseconds into `elapsed_ms`."""

    elapsed_ms: float = float("nan")

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def ms_per_item(elapsed_ms: float, n_items: int) -> float:
    """Average milliseconds per item (e.g. per window)."""
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    return elapsed_ms / n_items
