"""Average True Range (ATR) indicator for volatility measurement."""

from collections import deque
from typing import Optional

from .base import BaseIndicator


class ATR(BaseIndicator[float]):
    """
    Average True Range indicator.
    
    Measures market volatility by decomposing the entire range of an asset
    price for that period.
    """
    
    def __init__(self, period: int = 14):
        """Initialize ATR with period."""
        super().__init__(period)
        self._tr_values: deque[float] = deque(maxlen=period)
        self._prev_close: Optional[float] = None
        self._current_atr: Optional[float] = None
    
    def update_ohlc(
        self, 
        high: float, 
        low: float, 
        close: float
    ) -> Optional[float]:
        """
        Update with OHLC data.
        
        Args:
            high: High price
            low: Low price
            close: Close price
        
        Returns:
            ATR value if ready, None otherwise
        """
        # Calculate True Range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )
        
        self._prev_close = close
        self._tr_values.append(tr)
        
        if len(self._tr_values) < self.period:
            return None
        
        self._is_ready = True
        
        # Calculate ATR (Wilder's smoothing method)
        if self._current_atr is None:
            self._current_atr = sum(self._tr_values) / self.period
        else:
            self._current_atr = (
                (self._current_atr * (self.period - 1) + tr) / self.period
            )
        
        return self._current_atr
    
    def update(self, value: float) -> Optional[float]:
        """Not supported for ATR - use update_ohlc instead."""
        raise NotImplementedError("Use update_ohlc for ATR")
    
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._tr_values.clear()
        self._prev_close = None
        self._current_atr = None
