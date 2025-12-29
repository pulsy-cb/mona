"""Heuristic exit strategy based on volatility and momentum rules."""

from typing import Optional
from dataclasses import dataclass

from ..core.types import Signal, SignalType, Side
from ..core.config import TrailingStopConfig
from .base import BaseExitStrategy, ExitContext


@dataclass
class HeuristicConfig:
    """Configuration for heuristic exit rules."""
    # Base trailing stop config
    trailing: TrailingStopConfig
    
    # Volatility adjustments
    atr_multiplier_low_vol: float = 1.5  # Expand trailing in low volatility
    atr_multiplier_high_vol: float = 0.7  # Tighten in high volatility
    atr_threshold_low: float = 0.3  # ATR below this = low volatility  
    atr_threshold_high: float = 0.8  # ATR above this = high volatility
    
    # Momentum rules
    rsi_overbought: float = 70.0  # Consider exit for longs
    rsi_oversold: float = 30.0  # Consider exit for shorts
    momentum_reversal_threshold: float = -0.5  # Exit if momentum reverses
    
    # BB rules
    bb_exit_threshold_long: float = 0.95  # Exit long when %B > this
    bb_exit_threshold_short: float = 0.05  # Exit short when %B < this
    
    # Time rules
    min_holding_seconds: float = 60.0  # Don't exit before this
    stale_trade_seconds: float = 3600.0  # Force tighter stops after this


class HeuristicExit(BaseExitStrategy):
    """
    Heuristic exit strategy using volatility and momentum rules.
    
    Rules:
    1. Dynamic trailing stop based on ATR
    2. Exit on BB extremes (price near bands)
    3. Exit on momentum reversal
    4. Time-based stop tightening
    """
    
    def __init__(self, config: HeuristicConfig):
        """Initialize with heuristic configuration."""
        self.config = config
        self._highest_since_entry: Optional[float] = None
        self._lowest_since_entry: Optional[float] = None
        self._avg_atr: Optional[float] = None
    
    @property
    def name(self) -> str:
        return "heuristic"
    
    def _get_dynamic_trail_offset(self, context: ExitContext) -> float:
        """Calculate dynamic trailing offset based on volatility."""
        base_offset = self.config.trailing.trail_offset_points * 0.01
        
        if context.atr is None:
            return base_offset
        
        # Store average ATR for normalization
        if self._avg_atr is None:
            self._avg_atr = context.atr
        else:
            self._avg_atr = 0.95 * self._avg_atr + 0.05 * context.atr
        
        # Calculate relative volatility
        rel_vol = context.atr / self._avg_atr if self._avg_atr > 0 else 1.0
        
        if rel_vol < self.config.atr_threshold_low:
            # Low volatility - expand trailing to avoid premature exit
            return base_offset * self.config.atr_multiplier_low_vol
        elif rel_vol > self.config.atr_threshold_high:
            # High volatility - tighten trailing to protect profits
            return base_offset * self.config.atr_multiplier_high_vol
        
        return base_offset
    
    def _check_bb_exit(self, context: ExitContext) -> bool:
        """Check if should exit based on Bollinger Bands position."""
        if context.bb_percent_b is None:
            return False
        
        position = context.position
        
        if position.side == Side.LONG:
            # Exit long if price is near upper band
            if context.bb_percent_b > self.config.bb_exit_threshold_long:
                return True
        else:
            # Exit short if price is near lower band
            if context.bb_percent_b < self.config.bb_exit_threshold_short:
                return True
        
        return False
    
    def _check_momentum_reversal(self, context: ExitContext) -> bool:
        """Check if momentum is reversing against position."""
        if context.momentum is None:
            return False
        
        position = context.position
        
        if position.side == Side.LONG:
            # Exit long if momentum turns negative
            if context.momentum < self.config.momentum_reversal_threshold:
                return True
        else:
            # Exit short if momentum turns positive
            if context.momentum > -self.config.momentum_reversal_threshold:
                return True
        
        return False
    
    def _check_rsi_exit(self, context: ExitContext) -> bool:
        """Check if RSI indicates exit."""
        if context.rsi is None:
            return False
        
        position = context.position
        
        if position.side == Side.LONG:
            # Exit long if overbought
            if context.rsi > self.config.rsi_overbought:
                return True
        else:
            # Exit short if oversold
            if context.rsi < self.config.rsi_oversold:
                return True
        
        return False
    
    def should_exit(self, context: ExitContext) -> Optional[Signal]:
        """Determine if should exit using heuristic rules."""
        position = context.position
        price = context.current_price
        tick = context.current_tick
        
        # Don't exit too early
        if context.time_in_trade_seconds < self.config.min_holding_seconds:
            return None
        
        point = 0.01
        sl_distance = self.config.trailing.stop_loss_points * point
        
        # Dynamic trailing offset
        trail_offset = self._get_dynamic_trail_offset(context)
        
        # Tighten stop for stale trades
        if context.time_in_trade_seconds > self.config.stale_trade_seconds:
            trail_offset *= 0.7
        
        if position.side == Side.LONG:
            # Update highest price
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
            
            # Check dynamic trailing
            profit_points = (self._highest_since_entry - position.entry_price) / point
            if profit_points >= self.config.trailing.activation_points:
                trailing_sl = self._highest_since_entry - trail_offset
                if price <= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="dynamic_trailing"
                    )
            
            # Check heuristic exits only if in profit
            if context.unrealized_pnl_points > 0:
                if self._check_bb_exit(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="bb_extreme"
                    )
                
                if self._check_momentum_reversal(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="momentum_reversal"
                    )
                
                if self._check_rsi_exit(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="rsi_extreme"
                    )
        
        else:  # SHORT
            # Update lowest price
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
            
            # Check dynamic trailing
            profit_points = (position.entry_price - self._lowest_since_entry) / point
            if profit_points >= self.config.trailing.activation_points:
                trailing_sl = self._lowest_since_entry + trail_offset
                if price >= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="dynamic_trailing"
                    )
            
            # Check heuristic exits only if in profit
            if context.unrealized_pnl_points > 0:
                if self._check_bb_exit(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="bb_extreme"
                    )
                
                if self._check_momentum_reversal(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="momentum_reversal"
                    )
                
                if self._check_rsi_exit(context):
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="rsi_extreme"
                    )
        
        return None
    
    def reset(self) -> None:
        """Reset tracking variables."""
        self._highest_since_entry = None
        self._lowest_since_entry = None
