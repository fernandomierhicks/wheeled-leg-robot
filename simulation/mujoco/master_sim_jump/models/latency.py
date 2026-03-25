"""latency.py — Ring-buffer latency model for sensor and actuator delays.

Ported from latency_sensitivity/scenarios.py (inline deque pattern).
LatencyBuffer(n_steps=0) is a transparent pass-through — no conditional
branches needed in the sim loop.
"""
import collections
from typing import Any


class LatencyBuffer:
    """Fixed-depth ring buffer that delays values by n_steps control ticks.

    When ``n_steps=0``, push() immediately returns the pushed value (pass-through).
    When ``n_steps>0``, push() stores the new value and returns the oldest buffered value.

    Usage::
        buf = LatencyBuffer(n_steps=5, init_value=(0.0, 0.0, 0.0))
        delayed = buf.push(current_reading)

    Parameters
    ----------
    n_steps    : number of delay steps (0 = pass-through)
    init_value : value to pre-fill the buffer with
    """

    def __init__(self, n_steps: int, init_value: Any = 0.0) -> None:
        self._n = max(0, n_steps)
        if self._n == 0:
            # Pass-through mode — no buffer needed
            self._buf = None
            self._last = init_value
        else:
            self._buf = collections.deque(
                [init_value] * self._n, maxlen=self._n)
            self._last = init_value

    def push(self, value: Any) -> Any:
        """Store a new value and return the delayed (oldest) value."""
        if self._buf is None:
            # Pass-through: n_steps=0
            self._last = value
            return value
        oldest = self._buf[0]
        self._buf.append(value)
        self._last = oldest
        return oldest

    @property
    def oldest(self) -> Any:
        """The most recently returned delayed value (last push result)."""
        return self._last

    @property
    def n_steps(self) -> int:
        return self._n

    def reset(self, init_value: Any = None) -> None:
        """Re-fill buffer with init_value (or keep existing fill value)."""
        if init_value is None:
            init_value = self._last
        if self._buf is not None:
            self._buf.clear()
            for _ in range(self._n):
                self._buf.append(init_value)
        self._last = init_value
