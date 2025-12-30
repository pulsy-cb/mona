"""ML-based exit strategy using trained PPO model."""

from pathlib import Path
from typing import Optional

from ..core.types import Signal, SignalType
from ..core.config import TrailingStopConfig
from ..strategy.base import BaseExitStrategy, ExitContext
from .simple_features import SimpleFeatureExtractor


class MLExitStrategy(BaseExitStrategy):
    """
    Exit strategy using a trained RL model.
    
    Falls back to fixed stop loss if model not loaded.
    
    NOTE: Uses SimpleFeatureExtractor (9 features) to match PrecomputedTradingEnv.
    """
    
    def __init__(
        self,
        trailing_config: TrailingStopConfig,
        model_path: Optional[Path | str] = None,
        candle_only: bool = False
    ):
        """
        Initialize ML exit strategy.
        
        Args:
            trailing_config: Base trailing stop config (for fallback)
            model_path: Path to trained model
            candle_only: If True, only call ML model on candle closes (faster).
                         If False (default), call ML model on every tick.
        """
        self.trailing_config = trailing_config
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.candle_only = candle_only
        self.feature_extractor = SimpleFeatureExtractor(
            sl_points=trailing_config.stop_loss_points
        )
        
        self._highest_since_entry: Optional[float] = None
        self._lowest_since_entry: Optional[float] = None
        
        if self.model_path and self.model_path.exists():
            self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(self.model_path))
            print(f"Loaded ML model from {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
    @property
    def name(self) -> str:
        return "ml"
    
    def _check_stop_loss(self, context: ExitContext) -> bool:
        """Check if fixed stop loss is hit (safety net)."""
        from ..core.types import Side
        
        position = context.position
        price = context.current_price
        point = 0.01
        sl_distance = self.trailing_config.stop_loss_points * point
        
        if position.side == Side.LONG:
            return price <= position.entry_price - sl_distance
        else:
            return price >= position.entry_price + sl_distance
    
    def should_exit(self, context: ExitContext) -> Optional[Signal]:
        """
        Determine if should exit using ML model.
        
        If candle_only=True: Model is only called on candle closes (faster).
        If candle_only=False: Model is called on every tick (default, more accurate).
        """
        tick = context.current_tick
        
        # Always check stop loss first (safety) - runs every tick
        if self._check_stop_loss(context):
            return Signal(
                type=SignalType.EXIT,
                price=context.current_price,
                timestamp=tick.timestamp,
                reason="fixed_stop_loss"
            )
        
        # OPTIONAL OPTIMIZATION: Only call ML model on candle close
        if self.candle_only and context.current_candle is None:
            # Between candles - just check trailing stop as safety
            return self._check_trailing_stop(context)
        
        # If no model, use simple trailing stop
        if self.model is None:
            return self._fallback_trailing(context)
        
        # Get features and model prediction
        features = self.feature_extractor.extract(context)
        obs = features.to_array()
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        if action == 1:  # EXIT
            return Signal(
                type=SignalType.EXIT,
                price=context.current_price,
                timestamp=tick.timestamp,
                reason="ml_exit"
            )
        
        return None
    
    def _check_trailing_stop(self, context: ExitContext) -> Optional[Signal]:
        """Quick trailing stop check without ML - for between-candle ticks."""
        from ..core.types import Side
        
        position = context.position
        price = context.current_price
        tick = context.current_tick
        point = 0.01
        
        trail_activation = self.trailing_config.activation_points * point
        trail_offset = self.trailing_config.trail_offset_points * point
        
        if position.side == Side.LONG:
            if self._highest_since_entry is None:
                self._highest_since_entry = tick.bid
            else:
                self._highest_since_entry = max(self._highest_since_entry, tick.bid)
            
            profit = self._highest_since_entry - position.entry_price
            if profit >= trail_activation:
                trailing_sl = self._highest_since_entry - trail_offset
                if price <= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="trailing_stop"
                    )
        else:
            if self._lowest_since_entry is None:
                self._lowest_since_entry = tick.ask
            else:
                self._lowest_since_entry = min(self._lowest_since_entry, tick.ask)
            
            profit = position.entry_price - self._lowest_since_entry
            if profit >= trail_activation:
                trailing_sl = self._lowest_since_entry + trail_offset
                if price >= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="trailing_stop"
                    )
        
        return None
    
    def _fallback_trailing(self, context: ExitContext) -> Optional[Signal]:
        """Fallback to simple trailing stop."""
        from ..core.types import Side
        
        position = context.position
        price = context.current_price
        tick = context.current_tick
        point = 0.01
        
        trail_activation = self.trailing_config.activation_points * point
        trail_offset = self.trailing_config.trail_offset_points * point
        
        if position.side == Side.LONG:
            if self._highest_since_entry is None:
                self._highest_since_entry = tick.bid
            else:
                self._highest_since_entry = max(self._highest_since_entry, tick.bid)
            
            profit = self._highest_since_entry - position.entry_price
            if profit >= trail_activation:
                trailing_sl = self._highest_since_entry - trail_offset
                if price <= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="trailing_stop_fallback"
                    )
        else:
            if self._lowest_since_entry is None:
                self._lowest_since_entry = tick.ask
            else:
                self._lowest_since_entry = min(self._lowest_since_entry, tick.ask)
            
            profit = position.entry_price - self._lowest_since_entry
            if profit >= trail_activation:
                trailing_sl = self._lowest_since_entry + trail_offset
                if price >= trailing_sl:
                    return Signal(
                        type=SignalType.EXIT,
                        price=price,
                        timestamp=tick.timestamp,
                        reason="trailing_stop_fallback"
                    )
        
        return None
    
    def reset(self) -> None:
        """Reset tracking variables."""
        self._highest_since_entry = None
        self._lowest_since_entry = None
