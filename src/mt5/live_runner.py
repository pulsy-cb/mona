"""Live trading runner with ML exit strategy."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..core.types import Tick, Position, Signal, SignalType, Side
from ..core.config import Config
from ..data.converter import TickToOHLCConverter
from ..indicators import BollingerBands, ATR
from ..strategy.dark_venus import DarkVenusStrategy
from ..strategy.base import ExitContext
from ..ml.exit_strategy import MLExitStrategy
from .connector import MT5Connector, MT5Config

logger = logging.getLogger(__name__)


class LiveRunner:
    """
    Live trading runner.
    
    Executes Dark Venus entries with ML-optimized exits on MT5.
    """
    
    def __init__(
        self,
        mt5_config: MT5Config,
        config: Config,
        model_path: Optional[Path] = None,
        symbol: str = "XAUUSD"
    ):
        """
        Initialize live runner.
        
        Args:
            mt5_config: MT5 connection config
            config: Trading configuration
            model_path: Path to trained ML model
            symbol: Trading symbol
        """
        self.mt5_config = mt5_config
        self.config = config
        self.symbol = symbol
        
        # Components
        self.connector = MT5Connector(mt5_config)
        self.ohlc_converter = TickToOHLCConverter(config.timeframe_seconds)
        self.entry_strategy = DarkVenusStrategy(config)
        self.exit_strategy = MLExitStrategy(
            config.trailing,
            model_path=model_path
        )
        self.atr = ATR(period=14)
        
        # State
        self._position: Optional[Position] = None
        self._position_ticket: Optional[int] = None
        self._last_candle = None
        self._running = False
    
    def _on_tick(self, tick: Tick) -> None:
        """Process incoming tick."""
        try:
            # Update candle
            completed_candle = self.ohlc_converter.process_tick(tick)
            
            if completed_candle:
                self._last_candle = completed_candle
                self.atr.update_ohlc(
                    completed_candle.high,
                    completed_candle.low,
                    completed_candle.close
                )
                
                # Check entry on candle close
                if self._position is None:
                    signal = self.entry_strategy.update(completed_candle)
                    if signal:
                        self._open_position(signal, tick)
            
            # Check exit on every tick
            if self._position is not None:
                context = self._build_context(tick)
                exit_signal = self.exit_strategy.should_exit(context)
                if exit_signal:
                    self._close_position(exit_signal, tick)
        
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _build_context(self, tick: Tick) -> ExitContext:
        """Build exit context for current state."""
        bb_result = self.entry_strategy.last_bb_result
        
        return ExitContext(
            position=self._position,
            current_tick=tick,
            current_candle=self._last_candle,
            bb_upper=bb_result.upper if bb_result else None,
            bb_middle=bb_result.middle if bb_result else None,
            bb_lower=bb_result.lower if bb_result else None,
            bb_percent_b=bb_result.percent_b if bb_result else None,
            atr=self.atr._current_atr if self.atr.is_ready else None,
            rsi=None,
            momentum=None
        )
    
    def _open_position(self, signal: Signal, tick: Tick) -> None:
        """Open position via MT5."""
        if signal.type == SignalType.ENTRY_LONG:
            side = Side.LONG
            sl_price = tick.ask - (self.config.trailing.stop_loss_points * 0.01)
        else:
            side = Side.SHORT
            sl_price = tick.bid + (self.config.trailing.stop_loss_points * 0.01)
        
        ticket = self.connector.open_position(
            symbol=self.symbol,
            side=side,
            lot_size=self.config.trading.lot_size,
            sl_price=sl_price
        )
        
        if ticket:
            self._position_ticket = ticket
            self._position = Position(
                side=side,
                entry_price=tick.ask if side == Side.LONG else tick.bid,
                entry_time=tick.timestamp,
                size=self.config.trading.lot_size,
                stop_loss=sl_price
            )
            self.entry_strategy.set_has_position(True)
            self.exit_strategy.reset()
            logger.info(f"Opened {side.value} @ ticket {ticket}")
    
    def _close_position(self, signal: Signal, tick: Tick) -> None:
        """Close position via MT5."""
        if self._position is None or self._position_ticket is None:
            return
        
        success = self.connector.close_position(
            ticket=self._position_ticket,
            symbol=self.symbol,
            lot_size=self._position.size,
            side=self._position.side,
            comment=f"ml_exit_{signal.reason}"
        )
        
        if success:
            pnl_points = self._position.unrealized_pnl_points(signal.price)
            pnl = pnl_points * self._position.size * 100
            
            logger.info(
                f"Closed {self._position.side.value} @ {signal.price:.2f} "
                f"P&L: {pnl:.2f}€ ({signal.reason})"
            )
            
            self._position = None
            self._position_ticket = None
            self.entry_strategy.set_has_position(False)
    
    def run(self) -> None:
        """Start live trading loop."""
        if not self.connector.connect():
            logger.error("Failed to connect to MT5")
            return
        
        try:
            logger.info(f"Starting live trading on {self.symbol}...")
            self._running = True
            self.connector.subscribe_ticks(self.symbol, self._on_tick)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self._running = False
            self.connector.disconnect()
    
    def stop(self) -> None:
        """Stop trading loop."""
        self._running = False


def main():
    """CLI entry point for live trading."""
    parser = argparse.ArgumentParser(
        description="Run live trading with ML exit strategy"
    )
    parser.add_argument(
        "--login", type=int, required=True,
        help="MT5 account login"
    )
    parser.add_argument(
        "--password", type=str, required=True,
        help="MT5 account password"
    )
    parser.add_argument(
        "--server", type=str, required=True,
        help="MT5 server name"
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("models/best_model.zip"),
        help="Path to trained ML model"
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD",
        help="Trading symbol"
    )
    parser.add_argument(
        "--lot", type=float, default=0.01,
        help="Lot size"
    )
    parser.add_argument(
        "--sl", type=float, default=300,
        help="Stop loss in points"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configs
    mt5_config = MT5Config(
        login=args.login,
        password=args.password,
        server=args.server
    )
    
    config = Config.from_pine_params(sl_points=args.sl)
    config.trading.lot_size = args.lot
    
    # Run
    runner = LiveRunner(
        mt5_config=mt5_config,
        config=config,
        model_path=args.model,
        symbol=args.symbol
    )
    
    runner.run()


if __name__ == "__main__":
    main()
