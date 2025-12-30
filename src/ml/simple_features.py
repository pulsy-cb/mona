"""Simplified feature extractor compatible with PrecomputedTradingEnv."""

from dataclasses import dataclass
import numpy as np
from ..strategy.base import ExitContext


@dataclass
class SimpleFeatures:
    """Simplified features matching PrecomputedTradingEnv (9 features)."""
    
    # Static features (would be precomputed in training, but calculated here for backtest)
    bb_percent: float
    bb_width_norm: float
    atr_norm: float
    rsi_norm: float
    volatility_10: float
    velocity_5: float
    
    # Dynamic features (position-dependent)
    pnl_normalized: float
    time_in_trade: float
    distance_to_sl: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.bb_percent,
            self.bb_width_norm,
            self.atr_norm,
            self.rsi_norm,
            self.volatility_10,
            self.velocity_5,
            self.pnl_normalized,
            self.time_in_trade,
            self.distance_to_sl
        ], dtype=np.float32)
    
    @staticmethod
    def feature_dim() -> int:
        """Return feature dimension."""
        return 9


class SimpleFeatureExtractor:
    """
    Simplified feature extractor compatible with PrecomputedTradingEnv.
    
    Extracts only 9 features (6 static + 3 dynamic) to match the trained model.
    """
    
    def __init__(
        self,
        sl_points: float = 300,
        max_trade_time_seconds: float = 3600
    ):
        """
        Initialize simple feature extractor.
        
        Args:
            sl_points: Stop loss in points (for normalization)
            max_trade_time_seconds: Max expected trade duration
        """
        self.sl_points = sl_points
        self.max_trade_time = max_trade_time_seconds
        
        # Running averages for normalization (simplified)
        self._avg_atr = 1.0
        self._avg_bandwidth = 0.01
    
    def extract(self, context: ExitContext) -> SimpleFeatures:
        """
        Extract simplified features from exit context.
        
        Args:
            context: Current trading context
            
        Returns:
            SimpleFeatures with 9 features
        """
        position = context.position
        point = 0.01
        sl_distance = self.sl_points * point
        
        # === STATIC FEATURES (would be precomputed in training) ===
        
        # BB features
        bb_percent = context.bb_percent_b if context.bb_percent_b is not None else 0.5
        bb_percent = np.clip(bb_percent, -0.5, 1.5)
        
        bb_width_norm = 1.0
        if context.bb_upper is not None and context.bb_middle is not None:
            bandwidth = (context.bb_upper - context.bb_lower) / context.bb_middle
            bb_width_norm = bandwidth / self._avg_bandwidth
            bb_width_norm = np.clip(bb_width_norm, 0.2, 3.0)
            # Update running average
            self._avg_bandwidth = 0.99 * self._avg_bandwidth + 0.01 * bandwidth
        
        # ATR feature
        atr_norm = 1.0
        if context.atr is not None:
            atr_norm = context.atr / self._avg_atr
            atr_norm = np.clip(atr_norm, 0.2, 3.0)
            # Update running average
            self._avg_atr = 0.99 * self._avg_atr + 0.01 * context.atr
        
        # RSI feature
        rsi_norm = 0.0
        if context.rsi is not None:
            rsi_norm = (context.rsi - 50) / 50
            rsi_norm = np.clip(rsi_norm, -1.0, 1.0)
        
        # Volatility and velocity (simplified - set to 0 for backtest)
        # In training these are calculated from tick history
        volatility_10 = 0.0
        velocity_5 = 0.0
        
        # === DYNAMIC FEATURES (position-dependent) ===
        
        # PnL normalized
        pnl_points = context.unrealized_pnl_points
        pnl_normalized = np.clip(pnl_points / sl_distance, -2.0, 2.0)
        
        # Time in trade
        time_normalized = np.clip(
            context.time_in_trade_seconds / self.max_trade_time,
            0.0, 2.0
        )
        
        # Distance to SL
        current_price = context.current_price
        from ..core.types import Side
        
        if position.side == Side.LONG:
            distance_to_sl = (current_price - position.stop_loss) / sl_distance
        else:
            distance_to_sl = (position.stop_loss - current_price) / sl_distance
        distance_to_sl = np.clip(distance_to_sl, 0.0, 2.0)
        
        return SimpleFeatures(
            bb_percent=float(bb_percent),
            bb_width_norm=float(bb_width_norm),
            atr_norm=float(atr_norm),
            rsi_norm=float(rsi_norm),
            volatility_10=float(volatility_10),
            velocity_5=float(velocity_5),
            pnl_normalized=float(pnl_normalized),
            time_in_trade=float(time_normalized),
            distance_to_sl=float(distance_to_sl)
        )
