"""Numba-accelerated simulation core for ultra-fast backtesting."""

import numpy as np
from numba import njit


@njit(cache=True)
def simulate_trades_fast(
    bids: np.ndarray,
    asks: np.ndarray,
    tick_times: np.ndarray,  # int64, seconds since epoch
    candle_end_times: np.ndarray,  # int64, seconds since epoch
    long_signals: np.ndarray,  # bool
    short_signals: np.ndarray,  # bool
    bb_upper: np.ndarray,
    sl_distance: float,
    trail_activation: float,
    trail_offset: float,
    lot_size: float
) -> tuple:
    """
    Ultra-fast trade simulation using Numba JIT.
    
    Returns:
        Tuple of arrays: (entry_prices, exit_prices, entry_times, exit_times, 
                          pnls, is_long, exit_reasons)
        where exit_reasons: 0=stop_loss, 1=trailing_stop
    """
    n_ticks = len(bids)
    n_candles = len(candle_end_times)
    
    # Preallocate result arrays (max possible trades = n_candles)
    max_trades = n_candles
    entry_prices = np.zeros(max_trades, dtype=np.float64)
    exit_prices = np.zeros(max_trades, dtype=np.float64)
    entry_times = np.zeros(max_trades, dtype=np.int64)
    exit_times = np.zeros(max_trades, dtype=np.int64)
    pnls = np.zeros(max_trades, dtype=np.float64)
    is_long = np.zeros(max_trades, dtype=np.bool_)
    exit_reasons = np.zeros(max_trades, dtype=np.int32)
    
    trade_count = 0
    
    # State
    in_position = False
    position_is_long = False
    entry_price = 0.0
    entry_time = 0
    stop_loss = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    candle_idx = 0
    
    for i in range(n_ticks):
        bid = bids[i]
        ask = asks[i]
        tick_time = tick_times[i]
        
        # Update candle index
        while candle_idx < n_candles - 1 and tick_time >= candle_end_times[candle_idx]:
            candle_idx += 1
            
            # Check for entry on candle close
            if not in_position and candle_idx > 0:
                prev_idx = candle_idx - 1
                if not np.isnan(bb_upper[prev_idx]):
                    if long_signals[prev_idx]:
                        in_position = True
                        position_is_long = True
                        entry_price = ask
                        entry_time = tick_time
                        stop_loss = entry_price - sl_distance
                        highest_since_entry = bid
                        
                    elif short_signals[prev_idx]:
                        in_position = True
                        position_is_long = False
                        entry_price = bid
                        entry_time = tick_time
                        stop_loss = entry_price + sl_distance
                        lowest_since_entry = ask
        
        # Check exit conditions
        if in_position:
            if position_is_long:
                exit_price = bid
                if bid > highest_since_entry:
                    highest_since_entry = bid
                
                should_exit = False
                exit_reason = 0  # stop_loss
                
                if exit_price <= stop_loss:
                    should_exit = True
                else:
                    profit = highest_since_entry - entry_price
                    if profit >= trail_activation:
                        trailing_sl = highest_since_entry - trail_offset
                        if exit_price <= trailing_sl:
                            should_exit = True
                            exit_reason = 1  # trailing
                
                if should_exit:
                    pnl_points = exit_price - entry_price
                    pnl = pnl_points * lot_size * 100
                    
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = exit_price
                    entry_times[trade_count] = entry_time
                    exit_times[trade_count] = tick_time
                    pnls[trade_count] = pnl
                    is_long[trade_count] = True
                    exit_reasons[trade_count] = exit_reason
                    trade_count += 1
                    
                    in_position = False
            
            else:  # Short
                exit_price = ask
                if ask < lowest_since_entry:
                    lowest_since_entry = ask
                
                should_exit = False
                exit_reason = 0
                
                if exit_price >= stop_loss:
                    should_exit = True
                else:
                    profit = entry_price - lowest_since_entry
                    if profit >= trail_activation:
                        trailing_sl = lowest_since_entry + trail_offset
                        if exit_price >= trailing_sl:
                            should_exit = True
                            exit_reason = 1
                
                if should_exit:
                    pnl_points = entry_price - exit_price
                    pnl = pnl_points * lot_size * 100
                    
                    entry_prices[trade_count] = entry_price
                    exit_prices[trade_count] = exit_price
                    entry_times[trade_count] = entry_time
                    exit_times[trade_count] = tick_time
                    pnls[trade_count] = pnl
                    is_long[trade_count] = False
                    exit_reasons[trade_count] = exit_reason
                    trade_count += 1
                    
                    in_position = False
    
    # Return only used portion of arrays
    return (
        entry_prices[:trade_count],
        exit_prices[:trade_count],
        entry_times[:trade_count],
        exit_times[:trade_count],
        pnls[:trade_count],
        is_long[:trade_count],
        exit_reasons[:trade_count]
    )


@njit(cache=True)
def compute_bollinger_fast(
    closes: np.ndarray,
    period: int,
    deviation: float
) -> tuple:
    """Compute Bollinger Bands with Numba."""
    n = len(closes)
    bb_middle = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mean = 0.0
        for j in range(period):
            mean += window[j]
        mean /= period
        
        var = 0.0
        for j in range(period):
            var += (window[j] - mean) ** 2
        var /= period
        std = var ** 0.5
        
        bb_middle[i] = mean
        bb_upper[i] = mean + deviation * std
        bb_lower[i] = mean - deviation * std
    
    return bb_upper, bb_middle, bb_lower


@njit(cache=True)
def build_candles_fast(
    timestamps: np.ndarray,  # int64 seconds
    mids: np.ndarray,
    timeframe_seconds: int
) -> tuple:
    """Build OHLC candles from tick data with Numba."""
    n = len(timestamps)
    
    # First pass: count unique candles
    candle_starts = (timestamps // timeframe_seconds) * timeframe_seconds
    
    # Count candles
    n_candles = 1
    last_start = candle_starts[0]
    for i in range(1, n):
        if candle_starts[i] != last_start:
            n_candles += 1
            last_start = candle_starts[i]
    
    # Allocate arrays
    opens = np.zeros(n_candles, dtype=np.float64)
    highs = np.zeros(n_candles, dtype=np.float64)
    lows = np.zeros(n_candles, dtype=np.float64)
    closes = np.zeros(n_candles, dtype=np.float64)
    end_times = np.zeros(n_candles, dtype=np.int64)
    
    # Second pass: build candles
    candle_idx = 0
    candle_start = candle_starts[0]
    opens[0] = mids[0]
    highs[0] = mids[0]
    lows[0] = mids[0]
    
    for i in range(n):
        if candle_starts[i] != candle_start:
            # Finish previous candle
            closes[candle_idx] = mids[i - 1]
            end_times[candle_idx] = candle_start + timeframe_seconds
            
            # Start new candle
            candle_idx += 1
            candle_start = candle_starts[i]
            opens[candle_idx] = mids[i]
            highs[candle_idx] = mids[i]
            lows[candle_idx] = mids[i]
        else:
            if mids[i] > highs[candle_idx]:
                highs[candle_idx] = mids[i]
            if mids[i] < lows[candle_idx]:
                lows[candle_idx] = mids[i]
    
    # Finish last candle
    closes[candle_idx] = mids[n - 1]
    end_times[candle_idx] = candle_start + timeframe_seconds
    
    return opens, highs, lows, closes, end_times
