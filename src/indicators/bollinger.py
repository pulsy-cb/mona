"""Bollinger Bands indicator."""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import math

from .base import BaseIndicator


@dataclass(frozen=True, slots=True)
class BollingerBandsResult:
    """Bollinger Bands values at a point in time."""
    upper: float
    middle: float
    lower: float
    bandwidth: float  # (upper - lower) / middle
    percent_b: float  # (price - lower) / (upper - lower)
    
    @classmethod
    def from_values(
        cls, 
        upper: float, 
        middle: float, 
        lower: float,
        price: float
    ) -> "BollingerBandsResult":
        """Create result from band values and current price."""
        bandwidth = (upper - lower) / middle if middle != 0 else 0
        band_range = upper - lower
        percent_b = (price - lower) / band_range if band_range != 0 else 0.5
        return cls(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )


class BollingerBands(BaseIndicator[BollingerBandsResult]):
    """
    Bollinger Bands indicator.
    
    Matches Pine Script:
        [bb_middle, bb_upper, bb_lower] = ta.bb(bb_source, bb_period, bb_dev)
    """
    
    def __init__(self, period: int = 20, deviation: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: SMA lookback period
            deviation: Number of standard deviations
        """
        super().__init__(period)
        self.deviation = deviation
        self._prices: deque[float] = deque(maxlen=period)
        self._last_price: float = 0.0
    
    def update(self, price: float) -> Optional[BollingerBandsResult]:
        """
        Update with new price and return bands if ready.
        
        Args:
            price: Close price (or other source)
        
        Returns:
            BollingerBandsResult if enough data, None otherwise
        """
        self._prices.append(price)
        self._last_price = price
        
        if len(self._prices) < self.period:
            return None
        
        self._is_ready = True
        
        # Calculate SMA (middle band)
        prices_list = list(self._prices)
        sma = sum(prices_list) / self.period
        
        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in prices_list) / self.period
        std_dev = math.sqrt(variance)
        
        # Calculate bands
        upper = sma + (self.deviation * std_dev)
        lower = sma - (self.deviation * std_dev)
        
        return BollingerBandsResult.from_values(
            upper=upper,
            middle=sma,
            lower=lower,
            price=price
        )
    
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._prices.clear()
        self._last_price = 0.0
