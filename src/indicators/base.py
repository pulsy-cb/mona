"""Base class for indicators with rolling window."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Generic, TypeVar

T = TypeVar('T')


class BaseIndicator(ABC, Generic[T]):
    """Base class for all indicators."""
    
    def __init__(self, period: int):
        """Initialize with lookback period."""
        self.period = period
        self._values: deque[float] = deque(maxlen=period)
        self._is_ready = False
    
    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return self._is_ready
    
    @abstractmethod
    def update(self, value: float) -> Optional[T]:
        """Update indicator with new value and return result if ready."""
        pass
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._values.clear()
        self._is_ready = False
