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


class TradingEnv(gym.Env):
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
    
    metadata = {"render_modes": ["human"]}
    
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


class FastTradingEnv(gym.Env):
    """
    High-performance trading environment using NumPy arrays directly.
    
    This version avoids creating millions of Tick objects, providing
    5-10x speedup on large datasets (23M+ ticks).
    
    Actions:
        0 = HOLD (keep position open)
        1 = EXIT (close position)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        timestamps: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        config: Config,
        max_steps_per_episode: int = 100000
    ):
        """
        Initialize fast trading environment.
        
        Args:
            timestamps: int64 array of timestamps (seconds since epoch)
            bids: float32 array of bid prices
            asks: float32 array of ask prices
            config: Trading configuration
            max_steps_per_episode: Maximum ticks per episode
        """
        super().__init__()
        
        if gym is None:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")
        
        # Store arrays directly - no Tick object creation!
        self.timestamps = timestamps
        self.bids = bids
        self.asks = asks
        self.n_ticks = len(bids)
        
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
        self._episode_trades = []
        self._steps_in_trade = 0
        
        # Visibility: Episode history for debugging/analysis
        self.history = []
        
        # Dense reward tracking
        self._prev_unrealized_pnl = 0.0
        
        # Oracle system: Track optimal exit for efficiency calculation
        self._entry_idx = None
        self._max_potential_pnl = 0.0
    
    def _make_tick(self, idx: int) -> Tick:
        """Create a Tick object from arrays at given index (only when needed)."""
        from datetime import datetime
        return Tick(
            timestamp=datetime.fromtimestamp(int(self.timestamps[idx])),
            bid=float(self.bids[idx]),
            ask=float(self.asks[idx]),
            bid_volume=0.0,
            ask_volume=0.0
        )
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """Get observation for current state using array index."""
        from ..strategy.base import ExitContext
        
        if self._position is None:
            return np.zeros(Features.feature_dim(), dtype=np.float32)
        
        bb_result = self.entry_strategy.last_bb_result
        tick = self._make_tick(idx)
        
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
        if seed is not None:
            np.random.seed(seed)
        
        max_start = max(0, self.n_ticks - self.max_steps - 1000)
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
        
        # Reset visibility and oracle tracking
        self.history = []
        self._prev_unrealized_pnl = 0.0
        self._entry_idx = None
        self._max_potential_pnl = 0.0
        
        # Warm up indicators
        warmup_ticks = min(1000, self._tick_idx)
        for i in range(self._tick_idx - warmup_ticks, self._tick_idx):
            tick = self._make_tick(i)
            completed = self.ohlc_converter.process_tick(tick)
            if completed:
                self.atr.update_ohlc(completed.high, completed.low, completed.close)
                self.rsi.update(completed.close)
                self.momentum.update(completed.close)
                self.entry_strategy.update(completed)
                self._last_candle = completed
        
        # Find first position
        while self._position is None and self._tick_idx < self.n_ticks:
            tick = self._make_tick(self._tick_idx)
            completed = self.ohlc_converter.process_tick(tick)
            
            if completed:
                self._last_candle = completed
                self.atr.update_ohlc(completed.high, completed.low, completed.close)
                self.rsi.update(completed.close)
                self.momentum.update(completed.close)
                
                signal = self.entry_strategy.update(completed)
                if signal:
                    self._open_position(signal, self._tick_idx)
            
            self._tick_idx += 1
        
        obs = self._get_observation(min(self._tick_idx, self.n_ticks - 1))
        return obs, {}
    
    def _open_position(self, signal: Signal, idx: int) -> None:
        """Open position from entry signal."""
        from datetime import datetime
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        ts = datetime.fromtimestamp(int(self.timestamps[idx]))
        
        if signal.type == SignalType.ENTRY_LONG:
            entry_price = ask
            side = Side.LONG
            stop_loss = entry_price - (self.config.trailing.stop_loss_points * 0.01)
        else:
            entry_price = bid
            side = Side.SHORT
            stop_loss = entry_price + (self.config.trailing.stop_loss_points * 0.01)
        
        self._position = Position(
            side=side,
            entry_price=entry_price,
            entry_time=ts,
            size=self.config.trading.lot_size,
            stop_loss=stop_loss
        )
        self.entry_strategy.set_has_position(True)
        self._steps_in_trade = 0
        
        # Oracle: Track entry for efficiency calculation
        self._entry_idx = idx
        self._max_potential_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
    
    def _close_position(self, idx: int) -> float:
        """Close position and return P&L."""
        if self._position is None:
            return 0.0
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        
        if self._position.side == Side.LONG:
            exit_price = bid
        else:
            exit_price = ask
        
        pnl_points = self._position.unrealized_pnl_points(exit_price)
        pnl = pnl_points * self._position.size * 100
        
        self._episode_trades.append(pnl)
        self._position = None
        self.entry_strategy.set_has_position(False)
        
        return pnl
    
    def _check_stop_loss(self, idx: int) -> bool:
        """Check if stop loss is hit."""
        if self._position is None:
            return False
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        
        if self._position.side == Side.LONG:
            return bid <= self._position.stop_loss
        else:
            return ask >= self._position.stop_loss
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment with dense reward shaping."""
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        current_unrealized_pnl = 0.0
        
        if self._tick_idx >= self.n_ticks:
            terminated = True
            obs = np.zeros(Features.feature_dim(), dtype=np.float32)
            return obs, 0.0, terminated, truncated, info
        
        idx = self._tick_idx
        self._tick_idx += 1
        self._steps_in_trade += 1
        
        # Process tick for candle
        tick = self._make_tick(idx)
        completed = self.ohlc_converter.process_tick(tick)
        if completed:
            self._last_candle = completed
            self.atr.update_ohlc(completed.high, completed.low, completed.close)
            self.rsi.update(completed.close)
            self.momentum.update(completed.close)
        
        # Handle position with DENSE REWARD SHAPING
        if self._position is not None:
            # Get current price
            bid = float(self.bids[idx])
            ask = float(self.asks[idx])
            
            if self._position.side == Side.LONG:
                current_price = bid
            else:
                current_price = ask
            
            # Calculate current unrealized PnL
            current_unrealized_pnl = self._position.unrealized_pnl_points(current_price) * self._position.size * 100
            
            # ORACLE: Track maximum potential profit
            if self._entry_idx is not None:
                if self._position.side == Side.LONG:
                    max_price = float(np.max(self.bids[self._entry_idx:idx+1]))
                    potential = (max_price - self._position.entry_price) * self._position.size * 100
                else:
                    min_price = float(np.min(self.asks[self._entry_idx:idx+1]))
                    potential = (self._position.entry_price - min_price) * self._position.size * 100
                self._max_potential_pnl = max(self._max_potential_pnl, potential if potential > 0 else 0)
            
            # DENSE REWARD: Incremental change in PnL (what we gained/lost THIS tick)
            step_reward = current_unrealized_pnl - self._prev_unrealized_pnl
            self._prev_unrealized_pnl = current_unrealized_pnl
            
            # SMART TIME PENALTY: Only penalize when LOSING, reward patience when winning
            if current_unrealized_pnl > 0:
                time_reward = 0.001  # Small bonus for holding winners
            else:
                time_penalty = -0.002  # Lighter penalty (was -0.005)
                time_reward = time_penalty
            
            # MINIMUM HOLD TIME: Prevent panicking exits
            min_hold_steps = 50  # Must hold at least 50 ticks before allowing exit
            
            if self._check_stop_loss(idx):
                pnl = self._close_position(idx)
                # Penalize hitting stop loss HARD to encourage early exit
                reward = pnl - 5.0
                info['exit_reason'] = 'stop_loss'
                info['efficiency'] = 0.0  # Worst efficiency
            elif action == 1:  # EXIT
                # Block early exits - treat as HOLD if too early
                if self._steps_in_trade < min_hold_steps:
                    # Force HOLD, apply penalty for trying to exit too early
                    reward = step_reward + time_reward - 0.1  # Penalty for panic
                    info['blocked_exit'] = True
                else:
                    pnl = self._close_position(idx)
                    reward = pnl
                    
                    # ORACLE BONUS: Reward efficiency (capturing potential)
                    if self._max_potential_pnl > 0:
                        efficiency = max(0, pnl) / self._max_potential_pnl
                        efficiency_bonus = efficiency * 5.0  # Increased from 2.0
                        reward += efficiency_bonus
                        info['efficiency'] = efficiency
                    else:
                        info['efficiency'] = 1.0 if pnl >= 0 else 0.0
                    
                    # Bonus for patient exits (held longer)
                    patience_bonus = min(self._steps_in_trade / 500, 1.0)  # Up to +1 for holding 500 ticks
                    reward += patience_bonus
                    
                    info['exit_reason'] = 'agent_exit'
                    info['max_potential'] = self._max_potential_pnl
            else:  # HOLD
                # Dense reward: incremental PnL change + smart time reward
                reward = step_reward + time_reward
        
        # If not in position, find next entry
        if self._position is None and completed:
            signal = self.entry_strategy.update(completed)
            if signal:
                self._open_position(signal, idx)
        
        # Check termination
        if self._position is None and len(self._episode_trades) >= 20:
            terminated = True
        
        if self._tick_idx >= self.n_ticks - 1:
            truncated = True
        
        obs = self._get_observation(idx)
        info['trades'] = len(self._episode_trades)
        info['total_pnl'] = sum(self._episode_trades)
        
        # VISIBILITY: Record step in history
        self.history.append({
            'step': idx,
            'price': float(self.bids[idx]),
            'action': action,
            'reward': reward,
            'unrealized_pnl': current_unrealized_pnl,
            'is_in_position': 1 if self._position else 0,
            'exit_reason': info.get('exit_reason', None)
        })
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render current state - display last decision."""
        if len(self.history) > 0:
            h = self.history[-1]
            action_str = 'EXIT' if h['action'] == 1 else 'HOLD'
            pos_str = 'IN_POS' if h['is_in_position'] else 'NO_POS'
            exit_str = f" [{h['exit_reason']}]" if h['exit_reason'] else ""
            print(f"Step {h['step']:>7} | Price: {h['price']:.2f} | "
                  f"{pos_str} | Action: {action_str} | "
                  f"Reward: {h['reward']:+.4f} | PnL: {h['unrealized_pnl']:+.2f}{exit_str}")


class PrecomputedTradingEnv(gym.Env):
    """
    Ultra-fast trading environment using precomputed features.
    
    This version eliminates ALL per-step calculations:
    - No tick-to-candle conversion
    - No indicator updates (ATR, RSI, BB)
    - No feature extraction
    
    All static features are precomputed. Only position-dependent features
    (PnL, time in trade) are calculated at runtime.
    
    Expected speedup: 10-50x compared to FastTradingEnv.
    
    Actions:
        0 = HOLD (keep position open)
        1 = EXIT (close position)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        timestamps: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        precomputed_features: np.ndarray,
        feature_names: list[str],
        config: Config,
        max_steps_per_episode: int = 100000,
        use_entry_strategy: bool = True
    ):
        """
        Initialize precomputed trading environment.
        
        Args:
            timestamps: int64 array of timestamps (seconds since epoch)
            bids: float32 array of bid prices
            asks: float32 array of ask prices
            precomputed_features: float32 array of shape (N_ticks, N_features)
            feature_names: list of feature names
            config: Trading configuration
            max_steps_per_episode: Maximum ticks per episode
            use_entry_strategy: If True, use DarkVenus for entries. If False, simple random entries.
        """
        super().__init__()
        
        if gym is None:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")
        
        # Store arrays directly
        self.timestamps = timestamps
        self.bids = bids
        self.asks = asks
        self.precomputed_features = precomputed_features
        self.feature_names = feature_names
        self.n_ticks = len(bids)
        
        self.config = config
        self.max_steps = max_steps_per_episode
        self.use_entry_strategy = use_entry_strategy
        
        # Number of static (precomputed) and dynamic (runtime) features
        self.n_static_features = precomputed_features.shape[1]
        self.n_dynamic_features = 3  # PnL, time_in_trade, distance_to_sl
        total_features = self.n_static_features + self.n_dynamic_features
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # HOLD or EXIT
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # Entry strategy (optional - only if use_entry_strategy=True)
        if self.use_entry_strategy:
            self.ohlc_converter = TickToOHLCConverter(config.timeframe_seconds)
            self.entry_strategy = DarkVenusStrategy(config)
        else:
            self.ohlc_converter = None
            self.entry_strategy = None
        
        # State
        self._tick_idx = 0
        self._position: Optional[Position] = None
        self._episode_trades = []
        self._steps_in_trade = 0
        self._entry_idx = None
        
        # History for debugging
        self.history = []
        
        # Oracle tracking
        self._max_potential_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """Get observation by combining precomputed and dynamic features."""
        if self._position is None:
            # Not in position - return zeros
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # STATIC FEATURES: Simple array lookup (ultra-fast!)
        static_obs = self.precomputed_features[idx]
        
        # DYNAMIC FEATURES: Calculate based on current position
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        
        if self._position.side == Side.LONG:
            current_price = bid
        else:
            current_price = ask
        
        # PnL normalized by SL distance
        sl_distance = self.config.trailing.stop_loss_points * 0.01
        pnl_points = self._position.unrealized_pnl_points(current_price)
        pnl_norm = np.clip(pnl_points / sl_distance, -2.0, 2.0)
        
        # Time in trade (normalized)
        time_norm = np.clip(self._steps_in_trade / 3600, 0.0, 2.0)
        
        # Distance to SL (normalized)
        if self._position.side == Side.LONG:
            dist_to_sl = (current_price - self._position.stop_loss) / sl_distance
        else:
            dist_to_sl = (self._position.stop_loss - current_price) / sl_distance
        dist_to_sl = np.clip(dist_to_sl, 0.0, 2.0)
        
        dynamic_obs = np.array([pnl_norm, time_norm, dist_to_sl], dtype=np.float32)
        
        # Concatenate static + dynamic
        return np.concatenate([static_obs, dynamic_obs])
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        max_start = max(0, self.n_ticks - self.max_steps - 1000)
        self._tick_idx = np.random.randint(0, max(1, max_start))
        
        # Reset state
        self._position = None
        self._episode_trades = []
        self._steps_in_trade = 0
        self._entry_idx = None
        self.history = []
        self._max_potential_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
        
        # Reset entry strategy if used
        if self.use_entry_strategy:
            self.ohlc_converter.reset()
            self.entry_strategy.reset()
            
            # Warm up entry strategy
            warmup_ticks = min(1000, self._tick_idx)
            for i in range(self._tick_idx - warmup_ticks, self._tick_idx):
                tick = self._make_tick(i)
                completed = self.ohlc_converter.process_tick(tick)
                if completed:
                    self.entry_strategy.update(completed)
            
            # Find first position using strategy
            while self._position is None and self._tick_idx < self.n_ticks:
                tick = self._make_tick(self._tick_idx)
                completed = self.ohlc_converter.process_tick(tick)
                
                if completed:
                    signal = self.entry_strategy.update(completed)
                    if signal:
                        self._open_position(signal, self._tick_idx)
                
                self._tick_idx += 1
        else:
            # Simple random entry (for testing)
            self._open_random_position()
        
        obs = self._get_observation(min(self._tick_idx, self.n_ticks - 1))
        return obs, {}
    
    def _make_tick(self, idx: int) -> Tick:
        """Create a Tick object (only needed for entry strategy)."""
        from datetime import datetime
        return Tick(
            timestamp=datetime.fromtimestamp(int(self.timestamps[idx])),
            bid=float(self.bids[idx]),
            ask=float(self.asks[idx]),
            bid_volume=0.0,
            ask_volume=0.0
        )
    
    def _open_position(self, signal: Signal, idx: int) -> None:
        """Open position from entry signal."""
        from datetime import datetime
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        ts = datetime.fromtimestamp(int(self.timestamps[idx]))
        
        if signal.type == SignalType.ENTRY_LONG:
            entry_price = ask
            side = Side.LONG
            stop_loss = entry_price - (self.config.trailing.stop_loss_points * 0.01)
        else:
            entry_price = bid
            side = Side.SHORT
            stop_loss = entry_price + (self.config.trailing.stop_loss_points * 0.01)
        
        self._position = Position(
            side=side,
            entry_price=entry_price,
            entry_time=ts,
            size=self.config.trading.lot_size,
            stop_loss=stop_loss
        )
        
        if self.use_entry_strategy:
            self.entry_strategy.set_has_position(True)
        
        self._steps_in_trade = 0
        self._entry_idx = idx
        self._max_potential_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
    
    def _open_random_position(self) -> None:
        """Open a random position (for testing without entry strategy)."""
        from datetime import datetime
        
        idx = self._tick_idx
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        ts = datetime.fromtimestamp(int(self.timestamps[idx]))
        
        # Random long or short
        side = Side.LONG if np.random.rand() > 0.5 else Side.SHORT
        
        if side == Side.LONG:
            entry_price = ask
            stop_loss = entry_price - (self.config.trailing.stop_loss_points * 0.01)
        else:
            entry_price = bid
            stop_loss = entry_price + (self.config.trailing.stop_loss_points * 0.01)
        
        self._position = Position(
            side=side,
            entry_price=entry_price,
            entry_time=ts,
            size=self.config.trading.lot_size,
            stop_loss=stop_loss
        )
        
        self._steps_in_trade = 0
        self._entry_idx = idx
        self._max_potential_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
    
    def _close_position(self, idx: int) -> float:
        """Close position and return P&L."""
        if self._position is None:
            return 0.0
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        
        if self._position.side == Side.LONG:
            exit_price = bid
        else:
            exit_price = ask
        
        pnl_points = self._position.unrealized_pnl_points(exit_price)
        pnl = pnl_points * self._position.size * 100
        
        self._episode_trades.append(pnl)
        self._position = None
        
        if self.use_entry_strategy:
            self.entry_strategy.set_has_position(False)
        
        return pnl
    
    def _check_stop_loss(self, idx: int) -> bool:
        """Check if stop loss is hit."""
        if self._position is None:
            return False
        
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        
        if self._position.side == Side.LONG:
            return bid <= self._position.stop_loss
        else:
            return ask >= self._position.stop_loss
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step with precomputed features (ultra-fast!)."""
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        current_unrealized_pnl = 0.0
        
        if self._tick_idx >= self.n_ticks:
            terminated = True
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, 0.0, terminated, truncated, info
        
        idx = self._tick_idx
        self._tick_idx += 1
        self._steps_in_trade += 1
        
        # Handle position with dense reward shaping
        if self._position is not None:
            bid = float(self.bids[idx])
            ask = float(self.asks[idx])
            
            if self._position.side == Side.LONG:
                current_price = bid
            else:
                current_price = ask
            
            # Calculate current unrealized PnL
            current_unrealized_pnl = self._position.unrealized_pnl_points(current_price) * self._position.size * 100
            
            # Oracle: Track maximum potential profit
            if self._entry_idx is not None:
                if self._position.side == Side.LONG:
                    max_price = float(np.max(self.bids[self._entry_idx:idx+1]))
                    potential = (max_price - self._position.entry_price) * self._position.size * 100
                else:
                    min_price = float(np.min(self.asks[self._entry_idx:idx+1]))
                    potential = (self._position.entry_price - min_price) * self._position.size * 100
                self._max_potential_pnl = max(self._max_potential_pnl, potential if potential > 0 else 0)
            
            # Dense reward
            step_reward = current_unrealized_pnl - self._prev_unrealized_pnl
            self._prev_unrealized_pnl = current_unrealized_pnl
            
            # Smart time penalty
            if current_unrealized_pnl > 0:
                time_reward = 0.001
            else:
                time_reward = -0.002
            
            min_hold_steps = 50
            
            if self._check_stop_loss(idx):
                pnl = self._close_position(idx)
                reward = pnl - 5.0
                info['exit_reason'] = 'stop_loss'
                info['efficiency'] = 0.0
            elif action == 1:  # EXIT
                if self._steps_in_trade < min_hold_steps:
                    reward = step_reward + time_reward - 0.1
                    info['blocked_exit'] = True
                else:
                    pnl = self._close_position(idx)
                    reward = pnl
                    
                    if self._max_potential_pnl > 0:
                        efficiency = max(0, pnl) / self._max_potential_pnl
                        efficiency_bonus = efficiency * 5.0
                        reward += efficiency_bonus
                        info['efficiency'] = efficiency
                    else:
                        info['efficiency'] = 1.0 if pnl >= 0 else 0.0
                    
                    patience_bonus = min(self._steps_in_trade / 500, 1.0)
                    reward += patience_bonus
                    
                    info['exit_reason'] = 'agent_exit'
                    info['max_potential'] = self._max_potential_pnl
            else:  # HOLD
                reward = step_reward + time_reward
        
        # Find next entry if not in position
        if self._position is None:
            if self.use_entry_strategy:
                tick = self._make_tick(idx)
                completed = self.ohlc_converter.process_tick(tick)
                if completed:
                    signal = self.entry_strategy.update(completed)
                    if signal:
                        self._open_position(signal, idx)
            else:
                # Random entry
                if np.random.rand() < 0.01:  # 1% chance per tick
                    self._open_random_position()
        
        # Check termination
        if self._position is None and len(self._episode_trades) >= 20:
            terminated = True
        
        if self._tick_idx >= self.n_ticks - 1:
            truncated = True
        
        obs = self._get_observation(idx)
        info['trades'] = len(self._episode_trades)
        info['total_pnl'] = sum(self._episode_trades)
        
        # History tracking
        self.history.append({
            'step': idx,
            'price': float(self.bids[idx]),
            'action': action,
            'reward': reward,
            'unrealized_pnl': current_unrealized_pnl,
            'is_in_position': 1 if self._position else 0,
            'exit_reason': info.get('exit_reason', None)
        })
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render current state."""
        if len(self.history) > 0:
            h = self.history[-1]
            action_str = 'EXIT' if h['action'] == 1 else 'HOLD'
            pos_str = 'IN_POS' if h['is_in_position'] else 'NO_POS'
            exit_str = f" [{h['exit_reason']}]" if h['exit_reason'] else ""
            print(f"Step {h['step']:>7} | Price: {h['price']:.2f} | "
                  f"{pos_str} | Action: {action_str} | "
                  f"Reward: {h['reward']:+.4f} | PnL: {h['unrealized_pnl']:+.2f}{exit_str}")
