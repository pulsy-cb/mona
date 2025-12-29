"""Ultra-fast backtest engine using Numba JIT compilation."""

from datetime import datetime
from typing import Optional
import logging
import numpy as np
import pandas as pd

from ..core.types import Trade, Side
from ..core.config import Config
from ..data.loader import TickDataLoader, TickArray
from ..strategy.base import BaseExitStrategy
from .results import BacktestResults
from .numba_core import simulate_trades_fast, compute_bollinger_fast, build_candles_fast

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Ultra-fast tick-by-tick backtest engine.
    
    Uses Numba JIT compilation for ~10-50x speedup.
    """
    
    def __init__(
        self,
        config: Config,
        exit_strategy: BaseExitStrategy,
        initial_capital: float = 10000.0
    ):
        self.config = config
        self.exit_strategy = exit_strategy
        self.initial_capital = initial_capital
        
        # Pre-compute constants
        self.point = 0.01  # XAUUSD
        self.sl_distance = config.trailing.stop_loss_points * self.point
        self.trail_activation = config.trailing.activation_points * self.point
        self.trail_offset = config.trailing.trail_offset_points * self.point
        self.bb_period = config.bollinger.period
        self.bb_dev = config.bollinger.deviation
        self.timeframe_seconds = config.timeframe_seconds
        self.lot_size = config.trading.lot_size
        
    def run(self, loader: TickDataLoader, show_progress: bool = True) -> BacktestResults:
        """Run ultra-fast backtest with Numba."""
        arrays = loader.get_arrays()
        n_ticks = len(arrays.bids)
        
        logger.info(f"Running Numba-optimized backtest on {n_ticks:,} ticks...")
        
        # Pre-compute mid prices
        mids = (arrays.bids + arrays.asks) / 2
        
        # Convert timestamps to int64 seconds
        tick_times = arrays.timestamps.astype('datetime64[s]').astype(np.int64)
        
        # Build OHLC candles (Numba)
        logger.info("Building OHLC candles (Numba)...")
        opens, highs, lows, closes, candle_end_times = build_candles_fast(
            tick_times, mids, self.timeframe_seconds
        )
        n_candles = len(closes)
        logger.info(f"Built {n_candles:,} candles")
        
        # Compute Bollinger Bands (Numba)
        logger.info("Computing Bollinger Bands (Numba)...")
        bb_upper, bb_middle, bb_lower = compute_bollinger_fast(
            closes, self.bb_period, self.bb_dev
        )
        
        # Find entry signals
        can_long = self.config.trading.direction in ["long_only", "both"]
        can_short = self.config.trading.direction in ["short_only", "both"]
        
        if self.config.trading.bb_strategy == "sell_above_buy_below":
            long_signals = (closes < bb_lower) & can_long
            short_signals = (closes > bb_upper) & can_short
        else:
            long_signals = (closes > bb_upper) & can_long
            short_signals = (closes < bb_lower) & can_short
        
        # Handle NaN in signals (before BB warmup)
        long_signals = long_signals & ~np.isnan(bb_upper)
        short_signals = short_signals & ~np.isnan(bb_upper)
        
        # Simulate trades (Numba JIT - first call compiles, subsequent are fast)
        logger.info("Simulating trades (Numba JIT)...")
        (entry_prices, exit_prices, entry_times, exit_times, 
         pnls, is_long, exit_reasons) = simulate_trades_fast(
            arrays.bids,
            arrays.asks,
            tick_times,
            candle_end_times,
            long_signals,
            short_signals,
            bb_upper,
            self.sl_distance,
            self.trail_activation,
            self.trail_offset,
            self.lot_size
        )
        
        # Convert to Trade objects
        trades = []
        reason_map = {0: "fixed_stop_loss", 1: "trailing_stop"}
        
        for i in range(len(pnls)):
            trades.append(Trade(
                side=Side.LONG if is_long[i] else Side.SHORT,
                entry_price=entry_prices[i],
                exit_price=exit_prices[i],
                entry_time=datetime.fromtimestamp(entry_times[i]),
                exit_time=datetime.fromtimestamp(exit_times[i]),
                size=self.lot_size,
                pnl=pnls[i],
                pnl_points=(exit_prices[i] - entry_prices[i]) if is_long[i] else (entry_prices[i] - exit_prices[i]),
                exit_reason=reason_map.get(exit_reasons[i], "unknown")
            ))
        
        logger.info(f"Backtest complete. {len(trades)} trades executed.")
        
        return BacktestResults(trades=trades, initial_capital=self.initial_capital)
