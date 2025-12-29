"""Configuration management for the trading system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BollingerConfig:
    """Bollinger Bands configuration."""
    period: int = 20
    deviation: float = 2.0
    source: Literal["close", "open", "high", "low", "mid"] = "close"


@dataclass 
class TrailingStopConfig:
    """Trailing stop configuration."""
    enabled: bool = True
    stop_loss_points: float = 300.0  # Initial SL in points
    activation_points: float = 200.0  # Activate trailing after X points profit
    trail_offset_points: float = 100.0  # Distance to follow price


@dataclass
class TradingConfig:
    """Trading parameters."""
    direction: Literal["long_only", "short_only", "both"] = "both"
    bb_strategy: Literal["sell_above_buy_below", "buy_above_sell_below"] = "sell_above_buy_below"
    lot_size: float = 0.01
    point_value: float = 0.01  # Value of 1 point for the symbol


@dataclass
class MLConfig:
    """ML model configuration."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    model_path: Path = field(default_factory=lambda: Path("models/best_model.zip"))


@dataclass
class Config:
    """Main configuration container."""
    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    trailing: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    timeframe_seconds: int = 60  # M1 = 60 seconds
    symbol: str = "XAUUSD"
    
    @classmethod
    def from_pine_params(
        cls,
        sl_points: float = 300,
        trail_activation: float = 200,
        trail_offset: float = 100,
        bb_period: int = 20,
        bb_dev: float = 2.0
    ) -> "Config":
        """Create config from Pine Script parameters."""
        return cls(
            bollinger=BollingerConfig(period=bb_period, deviation=bb_dev),
            trailing=TrailingStopConfig(
                stop_loss_points=sl_points,
                activation_points=trail_activation,
                trail_offset_points=trail_offset
            )
        )
