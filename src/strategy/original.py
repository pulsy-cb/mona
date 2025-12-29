"""Original trailing stop strategy matching Pine Script behavior."""

from typing import Optional

from ..core.types import Signal, SignalType, Side
from ..core.config import TrailingStopConfig
from .base import BaseExitStrategy, ExitContext


class OriginalTrailingStop(BaseExitStrategy):
    """
    Original trailing stop strategy from Pine Script.
    
    Matches:
        strategy.exit("Exit Trailing", from_entry="Buy", 
            stop=strategy.position_avg_price - sl_val, 
            trail_points=trail_activation, 
            trail_offset=trail_offset)
    """
    
    def __init__(self, config: TrailingStopConfig):
        """Initialize with trailing stop configuration."""
        self.config = config
        self._highest_since_entry: Optional[float] = None
        self._lowest_since_entry: Optional[float] = None
    
    @property
    def name(self) -> str:
        return "original_trailing"
    
    def should_exit(self, context: ExitContext) -> Optional[Signal]:
        """Check if trailing stop or fixed stop is hit."""
        position = context.position
        price = context.current_price
        tick = context.current_tick
        
        # Point value (for XAUUSD, 1 point = 0.01)
        point = 0.01
        
        # Calculate SL in price terms
        sl_distance = self.config.stop_loss_points * point
        trail_activation = self.config.activation_points * point
        trail_offset = self.config.trail_offset_points * point
        
        if position.side == Side.LONG:
            # Update highest price since entry
            if self._highest_since_entry is None:
                self._highest_since_entry = tick.bid
            else:
                self._highest_since_entry = max(self._highest_since_entry, tick.bid)
            
            # Check fixed stop loss
            fixed_sl = position.entry_price - sl_distance
            if price <= fixed_sl:
                return Signal(
                    type=SignalType.EXIT,
                    price=price,
                    timestamp=tick.timestamp,
                    reason="fixed_stop_loss"
                )
            
            # Check trailing stop activation
            if self.config.enabled:
                profit_points = (self._highest_since_entry - position.entry_price) / point
                if profit_points >= self.config.activation_points:
                    trailing_sl = self._highest_since_entry - trail_offset
                    if price <= trailing_sl:
                        return Signal(
                            type=SignalType.EXIT,
                            price=price,
                            timestamp=tick.timestamp,
                            reason="trailing_stop"
                        )
        
        else:  # SHORT
            # Update lowest price since entry
            if self._lowest_since_entry is None:
                self._lowest_since_entry = tick.ask
            else:
                self._lowest_since_entry = min(self._lowest_since_entry, tick.ask)
            
            # Check fixed stop loss
            fixed_sl = position.entry_price + sl_distance
            if price >= fixed_sl:
                return Signal(
                    type=SignalType.EXIT,
                    price=price,
                    timestamp=tick.timestamp,
                    reason="fixed_stop_loss"
                )
            
            # Check trailing stop activation
            if self.config.enabled:
                profit_points = (position.entry_price - self._lowest_since_entry) / point
                if profit_points >= self.config.activation_points:
                    trailing_sl = self._lowest_since_entry + trail_offset
                    if price >= trailing_sl:
                        return Signal(
                            type=SignalType.EXIT,
                            price=price,
                            timestamp=tick.timestamp,
                            reason="trailing_stop"
                        )
        
        return None
    
    def reset(self) -> None:
        """Reset tracking variables for new position."""
        self._highest_since_entry = None
        self._lowest_since_entry = None
