"""Core data types for the trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class SignalType(Enum):
    """Signal types for entry/exit."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT = "exit"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Tick:
    """Single tick data point."""
    timestamp: datetime
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Current spread."""
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class OHLC:
    """OHLC candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_count: int = 0


@dataclass
class Position:
    """Active trading position."""
    side: Side
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    trailing_active: bool = False
    trailing_stop: Optional[float] = None
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L in points."""
        if self.side == Side.LONG:
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size
    
    def unrealized_pnl_points(self, current_price: float) -> float:
        """Calculate unrealized P&L in points (without size)."""
        if self.side == Side.LONG:
            return current_price - self.entry_price
        return self.entry_price - current_price


@dataclass(frozen=True, slots=True)
class Trade:
    """Completed trade record."""
    side: Side
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    pnl: float
    pnl_points: float
    exit_reason: str = "unknown"


@dataclass
class Signal:
    """Trading signal from strategy."""
    type: SignalType
    price: float
    timestamp: datetime
    reason: str = ""
    confidence: float = 1.0
