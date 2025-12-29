"""Dark Venus entry strategy - Bollinger Bands based entries."""

from typing import Optional

from ..core.types import OHLC, Signal, SignalType, Side
from ..core.config import Config
from ..indicators import BollingerBands, BollingerBandsResult


class DarkVenusStrategy:
    """
    Dark Venus entry strategy from Pine Script.
    
    Entry logic:
        if bb_strategy == "Sell Above and Buy Below":
            bb_buy = close < bb_lower
            bb_sell = close > bb_upper
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.bollinger = BollingerBands(
            period=config.bollinger.period,
            deviation=config.bollinger.deviation
        )
        self._last_bb_result: Optional[BollingerBandsResult] = None
        self._has_position = False
    
    def set_has_position(self, has_position: bool) -> None:
        """Update position state."""
        self._has_position = has_position
    
    def update(self, candle: OHLC) -> Optional[Signal]:
        """
        Update with new candle and check for entry signal.
        
        Args:
            candle: Completed OHLC candle
            
        Returns:
            Entry signal if conditions met, None otherwise
        """
        # Get source price based on config
        source = candle.close  # Default to close
        
        # Update Bollinger Bands
        bb_result = self.bollinger.update(source)
        if bb_result is None:
            return None
        
        self._last_bb_result = bb_result
        
        # Don't generate signals if already in position
        if self._has_position:
            return None
        
        # Check direction filter
        can_long = self.config.trading.direction in ["long_only", "both"]
        can_short = self.config.trading.direction in ["short_only", "both"]
        
        # Check BB strategy
        if self.config.trading.bb_strategy == "sell_above_buy_below":
            # Buy when close below lower band
            if candle.close < bb_result.lower and can_long:
                return Signal(
                    type=SignalType.ENTRY_LONG,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    reason="close_below_bb_lower"
                )
            
            # Sell when close above upper band
            if candle.close > bb_result.upper and can_short:
                return Signal(
                    type=SignalType.ENTRY_SHORT,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    reason="close_above_bb_upper"
                )
        
        else:  # buy_above_sell_below
            # Buy when close above upper band
            if candle.close > bb_result.upper and can_long:
                return Signal(
                    type=SignalType.ENTRY_LONG,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    reason="close_above_bb_upper"
                )
            
            # Sell when close below lower band
            if candle.close < bb_result.lower and can_short:
                return Signal(
                    type=SignalType.ENTRY_SHORT,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    reason="close_below_bb_lower"
                )
        
        return None
    
    @property
    def last_bb_result(self) -> Optional[BollingerBandsResult]:
        """Get last Bollinger Bands result for context."""
        return self._last_bb_result
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.bollinger.reset()
        self._last_bb_result = None
        self._has_position = False
