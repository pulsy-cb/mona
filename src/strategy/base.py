"""Base class for exit strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..core.types import Tick, OHLC, Position, Signal, SignalType


@dataclass
class ExitContext:
    """Context provided to exit strategy for decision making."""
    position: Position
    current_tick: Tick
    current_candle: Optional[OHLC]
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_percent_b: Optional[float] = None
    atr: Optional[float] = None
    rsi: Optional[float] = None
    momentum: Optional[float] = None
    
    @property
    def current_price(self) -> float:
        """Get current price for position evaluation."""
        from ..core.types import Side
        if self.position.side == Side.LONG:
            return self.current_tick.bid  # Sell at bid
        return self.current_tick.ask  # Buy to close at ask
    
    @property
    def unrealized_pnl_points(self) -> float:
        """Get unrealized P&L in points."""
        return self.position.unrealized_pnl_points(self.current_price)
    
    @property
    def time_in_trade_seconds(self) -> float:
        """Get time in trade in seconds."""
        delta = self.current_tick.timestamp - self.position.entry_time
        return delta.total_seconds()


class BaseExitStrategy(ABC):
    """Abstract base class for exit strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass
    
    @abstractmethod
    def should_exit(self, context: ExitContext) -> Optional[Signal]:
        """
        Determine if position should be exited.
        
        Args:
            context: Current market and position context
            
        Returns:
            Exit signal if should exit, None to hold
        """
        pass
    
    def reset(self) -> None:
        """Reset strategy state (if any)."""
        pass
