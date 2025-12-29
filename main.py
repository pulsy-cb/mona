"""Main entry point for the trading system."""

import sys
from pathlib import Path


def main():
    """Print usage information."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Trading Exit Optimizer - ML + Heuristic             ║
╚══════════════════════════════════════════════════════════════╝

Usage:

  1. BACKTEST avec stratégie originale (trailing stop fixe):
     python -m src.backtest.runner -d XAUUSD.parquet -s original

  2. BACKTEST avec stratégie heuristique (volatilité dynamique):
     python -m src.backtest.runner -d XAUUSD.parquet -s heuristic

  3. ENTRAÎNEMENT du modèle ML:
     python -m src.ml.train -d XAUUSD.parquet -o models/best_model.zip -t 100000

  4. BACKTEST avec modèle ML:
     python -m src.backtest.runner -d XAUUSD.parquet -s ml --model models/best_model.zip

  5. TRADING LIVE sur MT5:
     python -m src.mt5.live_runner --login YOUR_LOGIN --password YOUR_PASS --server YOUR_SERVER

Options courantes:
  --sl 300        Stop loss en points (défaut: 300)
  --activation 200  Activation trailing en points (défaut: 200)  
  --offset 100    Offset trailing en points (défaut: 100)
  -v              Mode verbose (debug)

Structure du projet:
  src/core/       Types et configuration
  src/data/       Chargement des données tick
  src/indicators/ Indicateurs techniques (BB, ATR, RSI)
  src/backtest/   Engine de backtest tick-by-tick
  src/strategy/   Stratégies de sortie (original, heuristic, ml)
  src/ml/         Environnement RL et entraînement
  src/mt5/        Connecteur MetaTrader 5
""")


if __name__ == "__main__":
    main()
