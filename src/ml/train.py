"""Training script for the ML exit optimization model."""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

from ..core.config import Config
from ..data.loader import TickDataLoader


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
    n_steps: int = 2048,
    batch_size: int = 64
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
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        raise ImportError(
            "stable-baselines3 not installed. Run:\n"
            "pip install stable-baselines3[extra]"
        )
    
    from .environment import TradingEnv
    
    logging.info("Loading tick data...")
    loader = TickDataLoader(data_path).load()
    ticks = list(loader)
    logging.info(f"Loaded {len(ticks):,} ticks")
    
    # Create config
    config = Config.from_pine_params(
        sl_points=300,
        trail_activation=200,
        trail_offset=100
    )
    
    # Split data for training and evaluation
    split_idx = int(len(ticks) * 0.8)
    train_ticks = ticks[:split_idx]
    eval_ticks = ticks[split_idx:]
    
    logging.info(f"Training ticks: {len(train_ticks):,}")
    logging.info(f"Evaluation ticks: {len(eval_ticks):,}")
    
    # Create environments
    def make_train_env():
        return TradingEnv(train_ticks, config)
    
    def make_eval_env():
        return TradingEnv(eval_ticks, config)
    
    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Create callbacks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_path.parent),
        name_prefix="ppo_exit_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path.parent),
        log_path=str(output_path.parent / "logs"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Create and train model
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
        tensorboard_log=str(output_path.parent / "tensorboard")
    )
    
    logging.info(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(str(output_path))
    logging.info(f"Model saved to {output_path}")
    
    # Quick evaluation
    logging.info("\nEvaluating final model...")
    env = TradingEnv(eval_ticks, config)
    obs, _ = env.reset()
    
    total_reward = 0
    trades = 0
    
    for _ in range(50000):  # Run for 50k steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            trades = info.get('trades', 0)
            break
    
    logging.info(f"Evaluation: {trades} trades, Total reward: {total_reward:.2f}")


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
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    train_model(
        data_path=args.data,
        output_path=args.output,
        total_timesteps=args.timesteps,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
