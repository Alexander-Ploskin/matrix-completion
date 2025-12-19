from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time


@dataclass
class Logger:
    """
    Collects per-iteration metrics + elapsed wall-clock seconds.

    Usage:
        logger = Logger()
        logger.start()
        ...
        logger.log(iter=i, rel_error=..., grad_norm=...)
    """
    meta: Dict[str, Any] = field(default_factory=dict)
    time_fn: Callable[[], float] = time.perf_counter

    _t0: Optional[float] = field(default=None, init=False, repr=False)
    records: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _default_iter: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        self.records.clear()
        self._default_iter = 0
        self._t0 = self.time_fn()

    def elapsed_s(self) -> float:
        if self._t0 is None:
            return 0.0
        return float(self.time_fn() - self._t0)

    def log(self, *, iter: Optional[int] = None, **metrics: Any) -> None:
        if iter is None:
            iter = self._default_iter
            self._default_iter += 1

        row = {"iter": int(iter), "time_s": self.elapsed_s(), **metrics, **self.meta}
        self.records.append(row)
