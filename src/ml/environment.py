"""Gymnasium environment for RL-based exit optimization."""

from typing import Optional, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from ..core.types import Tick, OHLC, Position, Signal, SignalType, Side
from ..core.config import Config
from ..data.converter import TickToOHLCConverter
from ..strategy.dark_venus import DarkVenusStrategy
from ..indicators import ATR, RSI, Momentum
from .features import FeatureExtractor, Features


class TradingEnv:
    """
    Trading environment for Reinforcement Learning.
    
    Simulates the trading loop where the agent only controls exit decisions.
    Entry decisions are made by the Dark Venus strategy.
    
    Actions:
        0 = HOLD (keep position open)
        1 = EXIT (close position)
    
    Reward:
        - Realized P&L on exit
        - Small negative reward for holding (time cost)
    """
    
    def __init__(
        self,
        ticks: list[Tick],
        config: Config,
        max_steps_per_episode: int = 100000
    ):
        """
        Initialize trading environment.
        
        Args:
            ticks: List of tick data for training
            config: Trading configuration
            max_steps_per_episode: Maximum ticks per episode
        """
        if gym is None:
            raise ImportError(
                "gymnasium not installed. Run: pip install gymnasium"
            )
        
        self.ticks = ticks
        self.config = config
        self.max_steps = max_steps_per_episode
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # HOLD or EXIT
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(Features.feature_dim(),),
            dtype=np.float32
        )
        
        # Components
        self.ohlc_converter = TickToOHLCConverter(config.timeframe_seconds)
        self.entry_strategy = DarkVenusStrategy(config)
        self.feature_extractor = FeatureExtractor(
            sl_points=config.trailing.stop_loss_points
        )
        
        # Indicators
        self.atr = ATR(period=14)
        self.rsi = RSI(period=14)
        self.momentum = Momentum(period=10)
        
        # State
        self._tick_idx = 0
        self._position: Optional[Position] = None
        self._last_candle: Optional[OHLC] = None
        self._last_bb_result = None
        self._episode_trades = []
        self._steps_in_trade = 0
    
    def _get_observation(self, tick: Tick) -> np.ndarray:
        """Get observation for current state."""
        from ..strategy.base import ExitContext
        
        if self._position is None:
            # Not in position - return zeros
            return np.zeros(Features.feature_dim(), dtype=np.float32)
        
        bb_result = self.entry_strategy.last_bb_result
        
        context = ExitContext(
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
        
        features = self.feature_extractor.extract(context)
        return features.to_array()
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment for new episode."""
        # Random start point in data
        if seed is not None:
            np.random.seed(seed)
        
        max_start = max(0, len(self.ticks) - self.max_steps - 1000)
        self._tick_idx = np.random.randint(0, max(1, max_start))
        
        # Reset components
        self.ohlc_converter.reset()
        self.entry_strategy.reset()
        self.atr.reset()
        self.rsi.reset()
        self.momentum.reset()
        
        # Reset state
        self._position = None
        self._last_candle = None
        self._episode_trades = []
        self._steps_in_trade = 0
        
        # Warm up indicators
        warmup_ticks = min(1000, self._tick_idx)
        for i in range(self._tick_idx - warmup_ticks, self._tick_idx):
            tick = self.ticks[i]
            completed = self.ohlc_converter.process_tick(tick)
            if completed:
                self.atr.update_ohlc(completed.high, completed.low, completed.close)
                self.rsi.update(completed.close)
                self.momentum.update(completed.close)
                self.entry_strategy.update(completed)
                self._last_candle = completed
        
        # Find first position
        while self._position is None and self._tick_idx < len(self.ticks):
            tick = self.ticks[self._tick_idx]
            completed = self.ohlc_converter.process_tick(tick)
            
            if completed:
                self._last_candle = completed
                self.atr.update_ohlc(completed.high, completed.low, completed.close)
                self.rsi.update(completed.close)
                self.momentum.update(completed.close)
                
                signal = self.entry_strategy.update(completed)
                if signal:
                    self._open_position(signal, tick)
            
            self._tick_idx += 1
        
        obs = self._get_observation(self.ticks[min(self._tick_idx, len(self.ticks)-1)])
        return obs, {}
    
    def _open_position(self, signal: Signal, tick: Tick) -> None:
        """Open position from entry signal."""
        if signal.type == SignalType.ENTRY_LONG:
            entry_price = tick.ask
            side = Side.LONG
            stop_loss = entry_price - (self.config.trailing.stop_loss_points * 0.01)
        else:
            entry_price = tick.bid
            side = Side.SHORT
            stop_loss = entry_price + (self.config.trailing.stop_loss_points * 0.01)
        
        self._position = Position(
            side=side,
            entry_price=entry_price,
            entry_time=tick.timestamp,
            size=self.config.trading.lot_size,
            stop_loss=stop_loss
        )
        self.entry_strategy.set_has_position(True)
        self._steps_in_trade = 0
    
    def _close_position(self, tick: Tick) -> float:
        """Close position and return P&L."""
        if self._position is None:
            return 0.0
        
        if self._position.side == Side.LONG:
            exit_price = tick.bid
        else:
            exit_price = tick.ask
        
        pnl_points = self._position.unrealized_pnl_points(exit_price)
        pnl = pnl_points * self._position.size * 100  # Approximate EUR
        
        self._episode_trades.append(pnl)
        self._position = None
        self.entry_strategy.set_has_position(False)
        
        return pnl
    
    def _check_stop_loss(self, tick: Tick) -> bool:
        """Check if stop loss is hit."""
        if self._position is None:
            return False
        
        if self._position.side == Side.LONG:
            return tick.bid <= self._position.stop_loss
        else:
            return tick.ask >= self._position.stop_loss
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = HOLD, 1 = EXIT
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if self._tick_idx >= len(self.ticks):
            terminated = True
            obs = np.zeros(Features.feature_dim(), dtype=np.float32)
            return obs, 0.0, terminated, truncated, info
        
        tick = self.ticks[self._tick_idx]
        self._tick_idx += 1
        self._steps_in_trade += 1
        
        # Process tick for candle
        completed = self.ohlc_converter.process_tick(tick)
        if completed:
            self._last_candle = completed
            self.atr.update_ohlc(completed.high, completed.low, completed.close)
            self.rsi.update(completed.close)
            self.momentum.update(completed.close)
        
        # Handle position
        if self._position is not None:
            # Check stop loss first
            if self._check_stop_loss(tick):
                pnl = self._close_position(tick)
                reward = pnl
                info['exit_reason'] = 'stop_loss'
            elif action == 1:  # EXIT
                pnl = self._close_position(tick)
                reward = pnl
                info['exit_reason'] = 'agent_exit'
            else:  # HOLD
                # Small time cost for holding
                reward = -0.001
        
        # If not in position, find next entry
        if self._position is None and completed:
            signal = self.entry_strategy.update(completed)
            if signal:
                self._open_position(signal, tick)
        
        # Check termination
        if self._position is None and len(self._episode_trades) >= 20:
            # Episode complete after 20 trades
            terminated = True
        
        if self._tick_idx >= len(self.ticks) - 1:
            truncated = True
        
        obs = self._get_observation(tick)
        info['trades'] = len(self._episode_trades)
        info['total_pnl'] = sum(self._episode_trades)
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render current state (not implemented)."""
        pass
