"""Analyze training performance from logs and evaluation results."""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_training_performance(models_dir: Path = Path("models")):
    """Analyze training performance from saved logs and models."""
    
    print("="*70)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("="*70)
    print()
    
    # 1. Analyze evaluation results
    eval_file = models_dir / "logs" / "evaluations.npz"
    if eval_file.exists():
        print("📊 EVALUATION RESULTS")
        print("-" * 70)
        data = np.load(eval_file)
        
        timesteps = data['timesteps']
        results = data['results']  # Shape: (n_evals, n_episodes)
        ep_lengths = data['ep_lengths']
        
        print(f"Number of evaluations: {len(timesteps)}")
        print(f"Timesteps evaluated: {timesteps}")
        print()
        
        for i, (ts, res, lengths) in enumerate(zip(timesteps, results, ep_lengths)):
            mean_reward = np.mean(res)
            std_reward = np.std(res)
            mean_length = np.mean(lengths)
            
            print(f"Evaluation {i+1} @ {ts:,} steps:")
            print(f"  Mean reward: {mean_reward:+.2f} ± {std_reward:.2f}")
            print(f"  Episode rewards: {res}")
            print(f"  Mean episode length: {mean_length:.0f} ticks")
            print()
    else:
        print("⚠️  No evaluation results found")
        print()
    
    # 2. List saved models
    print("💾 SAVED MODELS")
    print("-" * 70)
    model_files = sorted(models_dir.glob("*.zip"))
    
    if model_files:
        for model_file in model_files:
            size_mb = model_file.stat().st_size / 1024 / 1024
            print(f"  {model_file.name:<40} ({size_mb:.2f} MB)")
        print()
    else:
        print("  No models found")
        print()
    
    # 3. Check preprocessed data
    print("📦 PREPROCESSED DATA")
    print("-" * 70)
    cache_file = models_dir / "XAUUSD_preprocessed.npz"
    if cache_file.exists():
        size_mb = cache_file.stat().st_size / 1024 / 1024
        print(f"  Cache file: {cache_file.name}")
        print(f"  Size: {size_mb:.1f} MB")
        
        # Load to check contents
        data = np.load(cache_file, allow_pickle=True)
        print(f"  Ticks: {len(data['bids']):,}")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Feature names: {list(data['feature_names'])}")
        print()
    else:
        print("  No preprocessed data found")
        print()
    
    # 4. TensorBoard logs
    print("📈 TENSORBOARD LOGS")
    print("-" * 70)
    tb_dir = models_dir / "tensorboard"
    if tb_dir.exists():
        tb_runs = [d for d in tb_dir.iterdir() if d.is_dir()]
        print(f"  Number of runs: {len(tb_runs)}")
        print(f"  Runs: {[d.name for d in tb_runs]}")
        print()
        print("  To view in TensorBoard:")
        print(f"    tensorboard --logdir {tb_dir}")
        print()
    else:
        print("  No TensorBoard logs found")
        print()
    
    # 5. Performance summary
    print("📋 SUMMARY")
    print("-" * 70)
    
    if eval_file.exists():
        data = np.load(eval_file)
        final_results = data['results'][-1]
        final_mean = np.mean(final_results)
        
        print(f"Final evaluation mean reward: {final_mean:+.2f}")
        
        if final_mean < 0:
            print("⚠️  Model is currently losing money on average")
            print()
            print("RECOMMENDATIONS:")
            print("  1. Train for more timesteps (try 1M+ steps)")
            print("  2. Adjust PPO hyperparameters:")
            print("     - Increase ent_coef (e.g., 0.02) for more exploration")
            print("     - Adjust learning_rate (try 1e-4 or 5e-4)")
            print("  3. Review reward shaping in environment")
            print("  4. Check if entry strategy is generating good signals")
        else:
            print("✅ Model is profitable on average!")
            print()
            print("NEXT STEPS:")
            print("  1. Continue training for better performance")
            print("  2. Test on out-of-sample data")
            print("  3. Deploy to paper trading")
    
    print()
    print("="*70)


if __name__ == "__main__":
    analyze_training_performance()
