"""Momentum indicators (RSI, simple momentum)."""

from collections import deque
from typing import Optional

from .base import BaseIndicator


class RSI(BaseIndicator[float]):
    """
    Relative Strength Index indicator.
    
    Measures the speed and magnitude of price changes.
    """
    
    def __init__(self, period: int = 14):
        """Initialize RSI with period."""
        super().__init__(period)
        self._prev_price: Optional[float] = None
        self._gains: deque[float] = deque(maxlen=period)
        self._losses: deque[float] = deque(maxlen=period)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """
        Update with new price.
        
        Args:
            price: Current price
        
        Returns:
            RSI value (0-100) if ready, None otherwise
        """
        if self._prev_price is None:
            self._prev_price = price
            return None
        
        # Calculate change
        change = price - self._prev_price
        self._prev_price = price
        
        gain = max(0, change)
        loss = max(0, -change)
        
        self._gains.append(gain)
        self._losses.append(loss)
        
        if len(self._gains) < self.period:
            return None
        
        self._is_ready = True
        
        # Calculate average gain and loss (Wilder's smoothing)
        if self._avg_gain is None:
            self._avg_gain = sum(self._gains) / self.period
            self._avg_loss = sum(self._losses) / self.period
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        
        # Calculate RSI
        if self._avg_loss == 0:
            return 100.0
        
        rs = self._avg_gain / self._avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._prev_price = None
        self._gains.clear()
        self._losses.clear()
        self._avg_gain = None
        self._avg_loss = None


class Momentum(BaseIndicator[float]):
    """
    Simple momentum indicator (rate of change).
    
    Measures the difference between current price and price N periods ago.
    """
    
    def __init__(self, period: int = 10):
        """Initialize momentum with period."""
        super().__init__(period)
        self._prices: deque[float] = deque(maxlen=period + 1)
    
    def update(self, price: float) -> Optional[float]:
        """
        Update with new price.
        
        Args:
            price: Current price
        
        Returns:
            Momentum (price change) if ready, None otherwise
        """
        self._prices.append(price)
        
        if len(self._prices) <= self.period:
            return None
        
        self._is_ready = True
        
        # Return price difference
        return price - self._prices[0]
    
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._prices.clear()
