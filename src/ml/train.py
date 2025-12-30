"""Training script for the ML exit optimization model."""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np

from ..core.config import Config
from ..data.loader import TickDataLoader


def get_optimal_n_envs(requested: int | None = None) -> int:
    """
    Determine optimal number of parallel environments.
    
    Args:
        requested: User-requested number, or None for auto-detection
        
    Returns:
        Optimal number of environments (leaves 1-2 cores for system)
    """
    total_cpus = os.cpu_count() or 1
    
    if requested is not None:
        # User specified, but cap at total - 1
        return min(requested, max(1, total_cpus - 1))
    
    # Auto-detect: use half the cores (minimum 1, maximum total - 2)
    # This leaves room for the system and prevents overload
    if total_cpus <= 2:
        return 1
    elif total_cpus <= 4:
        return max(1, total_cpus - 1)  # Leave 1 core free
    else:
        return max(1, total_cpus - 2)  # Leave 2 cores free


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )



def train_model(
    data_path: Path,
    output_path: Path,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 256,  # Reduced from 2048 for faster iteration
    batch_size: int = 64,
    max_rows: int | None = None,
    n_envs: int | None = None
) -> None:
    """
    Train the PPO model for exit optimization.
    
    Args:
        data_path: Path to tick data
        output_path: Path to save model
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        raise ImportError(
            "stable-baselines3 not installed. Run:\n"
            "pip install stable-baselines3[extra]"
        )
    
    # Custom callback for better logging
    class LoggingCallback(BaseCallback):
        """Callback that provides clear status messages during training."""
        
        def __init__(self, eval_freq: int, verbose: int = 1):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.last_eval_step = 0
        
        def _on_step(self) -> bool:
            # Check if we're about to trigger evaluation
            if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                logging.info(f"\n🔄 Starting mid-training evaluation at step {self.num_timesteps:,}...")
                logging.info("   (This may take 1-2 minutes, please wait...)")
            return True
        
        def _on_rollout_end(self) -> None:
            # Log progress after each rollout
            pass
    
    from .environment import FastTradingEnv
    
    logging.info("Loading tick data...")
    loader = TickDataLoader(data_path, max_rows=max_rows).load()
    
    # Get numpy arrays directly - MUCH faster than creating Tick objects
    arrays = loader.get_arrays()
    total_ticks = len(arrays.bids)
    logging.info(f"Loaded {total_ticks:,} ticks as numpy arrays")
    
    # Create config
    config = Config.from_pine_params(
        sl_points=300,
        trail_activation=200,
        trail_offset=100
    )
    
    # Split data for training and evaluation (use indices, not copies)
    split_idx = int(total_ticks * 0.8)
    
    # Create tick lists lazily only for the environment
    logging.info(f"Training ticks: {split_idx:,}")
    logging.info(f"Evaluation ticks: {total_ticks - split_idx:,}")
    
    # Split arrays for train/eval (no object conversion needed!)
    train_timestamps = arrays.timestamps[:split_idx]
    train_bids = arrays.bids[:split_idx]
    train_asks = arrays.asks[:split_idx]
    
    eval_timestamps = arrays.timestamps[split_idx:]
    eval_bids = arrays.bids[split_idx:]
    eval_asks = arrays.asks[split_idx:]
    
    logging.info("Using FastTradingEnv - no Tick object conversion needed!")
    
    # Determine number of parallel environments
    actual_n_envs = get_optimal_n_envs(n_envs)
    logging.info(f"Using {actual_n_envs} parallel environments (detected {os.cpu_count()} CPUs)")
    
    # Create environment factory functions
    # Note: Each env gets the same data but will start at different random positions
    def make_train_env():
        return FastTradingEnv(train_timestamps, train_bids, train_asks, config)
    
    def make_eval_env():
        return FastTradingEnv(eval_timestamps, eval_bids, eval_asks, config)
    
    # Create vectorized environments
    if actual_n_envs > 1:
        try:
            # Use SubprocVecEnv for multi-core training
            train_env = SubprocVecEnv([make_train_env for _ in range(actual_n_envs)])
            eval_env = DummyVecEnv([make_eval_env])  # Eval uses single env
            logging.info(f"Successfully created {actual_n_envs} parallel training environments")
        except Exception as e:
            logging.warning(f"Failed to create SubprocVecEnv: {e}")
            logging.warning("Falling back to single environment (DummyVecEnv)")
            train_env = DummyVecEnv([make_train_env])
            eval_env = DummyVecEnv([make_eval_env])
    else:
        train_env = DummyVecEnv([make_train_env])
        eval_env = DummyVecEnv([make_eval_env])
    
    # Create callbacks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_path.parent),
        name_prefix="ppo_exit_model",
        verbose=1
    )
    
    # Evaluation frequency - balance between quality and speed
    eval_freq = max(10000, total_timesteps // 10)  # Evaluate ~10 times during training
    logging.info(f"Will evaluate every {eval_freq:,} timesteps")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path.parent),
        log_path=str(output_path.parent / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=2,  # 2 episodes for reliable evaluation
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Custom logging callback
    logging_callback = LoggingCallback(eval_freq=eval_freq)
    
    # Create and train model
    # For MLP policies, CPU is actually faster than GPU due to:
    # 1. Small network = minimal GPU benefit
    # 2. Data transfer overhead CPU <-> GPU
    # 3. Environment runs on CPU anyway
    # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
    import torch
    device = "cpu"  # Force CPU for MLP policy (faster than GPU for small networks)
    if torch.cuda.is_available():
        logging.info(f"ℹ️ GPU available ({torch.cuda.get_device_name(0)}) but using CPU for MLP policy (faster)")
    else:
        logging.info("ℹ️ Using CPU")
    
    logging.info("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device=device,  # Use GPU if available
        tensorboard_log=str(output_path.parent / "tensorboard")
    )
    
    logging.info(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, logging_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(str(output_path))
    logging.info(f"Model saved to {output_path}")
    
    # Quick evaluation - run multiple episodes for better statistics
    logging.info("\nEvaluating final model...")
    env = FastTradingEnv(eval_timestamps, eval_bids, eval_asks, config)
    
    n_eval_episodes = 5
    all_rewards = []
    all_trades = []
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=ep * 1000)  # Different starting points
        episode_reward = 0
        episode_trades = 0
        
        for step in range(100000):  # Max 100k steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                episode_trades = info.get('trades', 0)
                break
        
        all_rewards.append(episode_reward)
        all_trades.append(episode_trades)
        logging.info(f"  Episode {ep+1}: {episode_trades} trades, reward: {episode_reward:.2f}")
    
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_trades = sum(all_trades) / len(all_trades)
    logging.info(f"\nEvaluation Summary ({n_eval_episodes} episodes):")
    logging.info(f"  Average trades: {avg_trades:.1f}")
    logging.info(f"  Average reward: {avg_reward:.2f}")
    logging.info(f"  Total trades: {sum(all_trades)}")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train ML exit optimization model"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("XAUUSD.parquet"),
        help="Path to tick data"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("models/best_model.zip"),
        help="Output path for model"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--max-rows", "-m",
        type=int,
        default=None,
        help="Maximum rows to load (for faster testing). Recommended: 1000000 for quick tests"
    )
    parser.add_argument(
        "--n-envs", "-n",
        type=int,
        default=None,
        help="Number of parallel environments. Default: auto-detect (uses available CPUs - 2)"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    train_model(
        data_path=args.data,
        output_path=args.output,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        max_rows=args.max_rows,
        n_envs=args.n_envs
    )


if __name__ == "__main__":
    main()
