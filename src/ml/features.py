"""Feature extraction for ML model."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..strategy.base import ExitContext


@dataclass  
class Features:
    """Normalized features for ML model."""
    
    # Position features
    pnl_normalized: float  # Unrealized P&L / SL distance
    time_in_trade: float  # Time in trade / max expected time
    
    # Bollinger Bands features
    bb_percent_b: float  # Position within bands (0-1)
    bb_bandwidth_normalized: float  # Current bandwidth / average
    
    # Volatility features
    atr_normalized: float  # Current ATR / average ATR
    
    # Momentum features  
    rsi_normalized: float  # (RSI - 50) / 50, range -1 to 1
    momentum_normalized: float  # Momentum / ATR
    
    # Price features
    distance_to_sl: float  # Distance to SL / SL distance (0-1)
    distance_to_entry: float  # Signed distance to entry / SL
    is_in_profit: float  # 1.0 if in profit, 0.0 otherwise
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.pnl_normalized,
            self.time_in_trade,
            self.bb_percent_b,
            self.bb_bandwidth_normalized,
            self.atr_normalized,
            self.rsi_normalized,
            self.momentum_normalized,
            self.distance_to_sl,
            self.distance_to_entry,
            self.is_in_profit
        ], dtype=np.float32)
    
    @staticmethod
    def feature_dim() -> int:
        """Return feature dimension."""
        return 10


class FeatureExtractor:
    """Extract and normalize features from trading context."""
    
    def __init__(
        self,
        sl_points: float = 300,
        max_trade_time_seconds: float = 3600,
        avg_atr: Optional[float] = None,
        avg_bandwidth: Optional[float] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            sl_points: Stop loss in points (for normalization)
            max_trade_time_seconds: Max expected trade duration
            avg_atr: Average ATR for normalization (learned online)
            avg_bandwidth: Average BB bandwidth (learned online)
        """
        self.sl_points = sl_points
        self.max_trade_time = max_trade_time_seconds
        self._avg_atr = avg_atr or 1.0
        self._avg_bandwidth = avg_bandwidth or 0.01
        self._atr_ema_alpha = 0.01
        self._bandwidth_ema_alpha = 0.01
    
    def _update_running_averages(self, context: ExitContext) -> None:
        """Update running averages for normalization."""
        if context.atr is not None:
            self._avg_atr = (
                self._atr_ema_alpha * context.atr + 
                (1 - self._atr_ema_alpha) * self._avg_atr
            )
        
        if context.bb_upper is not None and context.bb_middle is not None:
            bandwidth = (context.bb_upper - context.bb_lower) / context.bb_middle
            self._avg_bandwidth = (
                self._bandwidth_ema_alpha * bandwidth +
                (1 - self._bandwidth_ema_alpha) * self._avg_bandwidth
            )
    
    def extract(self, context: ExitContext) -> Features:
        """
        Extract normalized features from exit context.
        
        Args:
            context: Current trading context
            
        Returns:
            Normalized features for model
        """
        self._update_running_averages(context)
        
        position = context.position
        point = 0.01  # For XAUUSD
        sl_distance = self.sl_points * point
        
        # P&L features
        pnl_points = context.unrealized_pnl_points
        pnl_normalized = np.clip(pnl_points / sl_distance, -2.0, 2.0)
        is_in_profit = 1.0 if pnl_points > 0 else 0.0
        
        # Time feature
        time_normalized = np.clip(
            context.time_in_trade_seconds / self.max_trade_time, 
            0.0, 2.0
        )
        
        # BB features
        bb_percent_b = context.bb_percent_b if context.bb_percent_b is not None else 0.5
        bb_percent_b = np.clip(bb_percent_b, -0.5, 1.5)
        
        bandwidth_normalized = 1.0
        if context.bb_upper is not None and context.bb_middle is not None:
            bandwidth = (context.bb_upper - context.bb_lower) / context.bb_middle
            bandwidth_normalized = bandwidth / self._avg_bandwidth
            bandwidth_normalized = np.clip(bandwidth_normalized, 0.2, 3.0)
        
        # ATR feature
        atr_normalized = 1.0
        if context.atr is not None:
            atr_normalized = context.atr / self._avg_atr
            atr_normalized = np.clip(atr_normalized, 0.2, 3.0)
        
        # RSI feature (normalized to -1 to 1)
        rsi_normalized = 0.0
        if context.rsi is not None:
            rsi_normalized = (context.rsi - 50) / 50
            rsi_normalized = np.clip(rsi_normalized, -1.0, 1.0)
        
        # Momentum feature
        momentum_normalized = 0.0
        if context.momentum is not None and context.atr is not None:
            momentum_normalized = context.momentum / context.atr
            momentum_normalized = np.clip(momentum_normalized, -3.0, 3.0)
        
        # Distance features
        current_price = context.current_price
        from ..core.types import Side
        
        if position.side == Side.LONG:
            distance_to_sl = (current_price - position.stop_loss) / sl_distance
        else:
            distance_to_sl = (position.stop_loss - current_price) / sl_distance
        distance_to_sl = np.clip(distance_to_sl, 0.0, 2.0)
        
        distance_to_entry = pnl_normalized  # Already normalized
        
        return Features(
            pnl_normalized=float(pnl_normalized),
            time_in_trade=float(time_normalized),
            bb_percent_b=float(bb_percent_b),
            bb_bandwidth_normalized=float(bandwidth_normalized),
            atr_normalized=float(atr_normalized),
            rsi_normalized=float(rsi_normalized),
            momentum_normalized=float(momentum_normalized),
            distance_to_sl=float(distance_to_sl),
            distance_to_entry=float(distance_to_entry),
            is_in_profit=float(is_in_profit)
        )
