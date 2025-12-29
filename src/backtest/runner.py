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
    sl_points: float = 300,
    trail_activation: float = 200,
    trail_offset: float = 100,
    model_path: str | None = None
) -> BacktestResults:
    """Run backtest with specified parameters."""
    # Create config
    config = Config.from_pine_params(
        sl_points=sl_points,
        trail_activation=trail_activation,
        trail_offset=trail_offset
    )
    
    # Load tick data
    logging.info(f"Loading tick data from {data_path}...")
    loader = TickDataLoader(data_path).load()
    logging.info(f"Loaded {len(loader):,} ticks")
    logging.info(f"Date range: {loader.date_range[0]} to {loader.date_range[1]}")
    
    # Create exit strategy (for reference, engine uses its own fast logic)
    exit_strategy = OriginalTrailingStop(config.trailing)
    
    # Create and run engine
    engine = BacktestEngine(config, exit_strategy)
    
    logging.info(f"Running backtest with {strategy_name} exit strategy...")
    results = engine.run(loader)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run tick-by-tick backtest")
    parser.add_argument("--data", "-d", type=Path, default=Path("XAUUSD.parquet"))
    parser.add_argument("--strategy", "-s", choices=["original", "heuristic", "ml"], default="original")
    parser.add_argument("--sl", type=float, default=300)
    parser.add_argument("--activation", type=float, default=200)
    parser.add_argument("--offset", type=float, default=100)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    results = run_backtest(
        data_path=args.data,
        strategy_name=args.strategy,
        sl_points=args.sl,
        trail_activation=args.activation,
        trail_offset=args.offset,
        model_path=args.model
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
