"""Convert ticks to OHLC candles."""

from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Iterator

from ..core.types import Tick, OHLC


class TickToOHLCConverter:
    """Convert streaming ticks to OHLC candles."""
    
    def __init__(self, timeframe_seconds: int = 60):
        """Initialize converter with timeframe in seconds."""
        self.timeframe_seconds = timeframe_seconds
        self._current_candle: Optional[dict] = None
        self._candle_end_time: Optional[datetime] = None
        
    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the candle containing this timestamp."""
        # Align to timeframe boundaries
        total_seconds = int(timestamp.timestamp())
        candle_start_seconds = (total_seconds // self.timeframe_seconds) * self.timeframe_seconds
        return datetime.fromtimestamp(candle_start_seconds)
    
    def process_tick(self, tick: Tick) -> Optional[OHLC]:
        """
        Process a single tick and return completed candle if any.
        
        Returns:
            OHLC candle if a candle was completed, None otherwise.
        """
        price = tick.mid  # Use mid price for OHLC
        
        # First tick - initialize candle
        if self._current_candle is None:
            candle_start = self._get_candle_start(tick.timestamp)
            self._candle_end_time = candle_start + timedelta(seconds=self.timeframe_seconds)
            self._current_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': tick.bid_volume + tick.ask_volume,
                'tick_count': 1
            }
            return None
        
        # Check if we've moved to a new candle
        if tick.timestamp >= self._candle_end_time:
            # Complete the current candle
            completed = OHLC(
                timestamp=self._current_candle['timestamp'],
                open=self._current_candle['open'],
                high=self._current_candle['high'],
                low=self._current_candle['low'],
                close=self._current_candle['close'],
                volume=self._current_candle['volume'],
                tick_count=self._current_candle['tick_count']
            )
            
            # Start new candle
            candle_start = self._get_candle_start(tick.timestamp)
            self._candle_end_time = candle_start + timedelta(seconds=self.timeframe_seconds)
            self._current_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': tick.bid_volume + tick.ask_volume,
                'tick_count': 1
            }
            
            return completed
        
        # Update current candle
        self._current_candle['high'] = max(self._current_candle['high'], price)
        self._current_candle['low'] = min(self._current_candle['low'], price)
        self._current_candle['close'] = price
        self._current_candle['volume'] += tick.bid_volume + tick.ask_volume
        self._current_candle['tick_count'] += 1
        
        return None
    
    def get_current_candle(self) -> Optional[OHLC]:
        """Get the current incomplete candle."""
        if self._current_candle is None:
            return None
        return OHLC(
            timestamp=self._current_candle['timestamp'],
            open=self._current_candle['open'],
            high=self._current_candle['high'],
            low=self._current_candle['low'],
            close=self._current_candle['close'],
            volume=self._current_candle['volume'],
            tick_count=self._current_candle['tick_count']
        )
    
    def flush(self) -> Optional[OHLC]:
        """Flush and return the current incomplete candle."""
        candle = self.get_current_candle()
        self._current_candle = None
        self._candle_end_time = None
        return candle
    
    def reset(self) -> None:
        """Reset the converter state."""
        self._current_candle = None
        self._candle_end_time = None


def convert_ticks_to_ohlc(
    ticks: Iterator[Tick],
    timeframe_seconds: int = 60
) -> list[OHLC]:
    """
    Convert an iterator of ticks to a list of OHLC candles.
    
    Args:
        ticks: Iterator of Tick objects
        timeframe_seconds: Candle timeframe in seconds (60 = M1)
    
    Returns:
        List of completed OHLC candles
    """
    converter = TickToOHLCConverter(timeframe_seconds)
    candles = []
    
    for tick in ticks:
        completed = converter.process_tick(tick)
        if completed:
            candles.append(completed)
    
    # Flush last incomplete candle
    last_candle = converter.flush()
    if last_candle:
        candles.append(last_candle)
    
    return candles
