"""Backtest runner - CLI interface for running backtests."""

import argparse
import logging
from pathlib import Path

from ..core.config import Config, TrailingStopConfig
from ..data.loader import TickDataLoader
from ..strategy.original import OriginalTrailingStop
from ..strategy.heuristic import HeuristicExit, HeuristicConfig
from .engine import BacktestEngine
from .results import BacktestResults


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_backtest(
    data_path: Path,
    strategy_name: str = "original",
    sl_points: float = 200,
    trail_activation: float = 50,
    trail_offset: float = 10,
    model_path: str | None = None,
    sample_ratio: float = 1.0,
    max_rows: int | None = None
) -> BacktestResults:
    """Run backtest with specified parameters."""
    # Create config
    config = Config.from_pine_params(
        sl_points=sl_points,
        trail_activation=trail_activation,
        trail_offset=trail_offset
    )
    
    # Load tick data with memory optimization
    logging.info(f"Loading tick data from {data_path}...")
    loader = TickDataLoader(
        data_path, 
        sample_ratio=sample_ratio,
        max_rows=max_rows
    ).load()
    logging.info(f"Loaded {len(loader):,} ticks")
    logging.info(f"Date range: {loader.date_range[0]} to {loader.date_range[1]}")
    
    # Create exit strategy
    use_numba = False
    
    if strategy_name == "heuristic":
        # Create config for heuristic
        heuristic_config = HeuristicConfig(
            trailing=config.trailing,
            # Using default params for now, could be exposed to CLI
        )
        exit_strategy = HeuristicExit(heuristic_config)
        use_numba = False  # Heuristic requires Python engine for complex logic
    else:
        # Original strategy
        exit_strategy = OriginalTrailingStop(config.trailing)
        use_numba = True  # Original logic is supported by Numba engine
    
    # Create and run engine
    engine = BacktestEngine(config, exit_strategy)
    
    if use_numba:
        logging.info(f"Running Numba-optimized backtest with {strategy_name} strategy...")
        results = engine.run(loader)
    else:
        logging.info(f"Running Python-based backtest with {strategy_name} strategy (slower)...")
        results = engine.run_python(loader)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run tick-by-tick backtest")
    parser.add_argument("--data", "-d", type=Path, default=Path("XAUUSD.parquet"))
    parser.add_argument("--strategy", "-s", choices=["original", "heuristic", "ml"], default="original")
    parser.add_argument("--sl", type=float, default=200)
    parser.add_argument("--activation", type=float, default=50)
    parser.add_argument("--offset", type=float, default=10)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    # Memory optimization options
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Sample ratio (0.0-1.0). Use 0.1 for 10%% of data. Default: 1.0 (all)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to load. E.g., 5000000 for 5M ticks.")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    results = run_backtest(
        data_path=args.data,
        strategy_name=args.strategy,
        sl_points=args.sl,
        trail_activation=args.activation,
        trail_offset=args.offset,
        model_path=args.model,
        sample_ratio=args.sample,
        max_rows=args.max_rows
    )
    
    results.print_summary()
    
    # Exit reasons breakdown
    exit_reasons = {}
    for trade in results.trades:
        reason = trade.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0.0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += trade.pnl
    
    print("\nExit Reasons Breakdown:")
    print("-" * 40)
    for reason, stats in sorted(exit_reasons.items()):
        print(f"  {reason}: {stats['count']} trades, {stats['pnl']:.2f}€")


if __name__ == "__main__":
    main()
