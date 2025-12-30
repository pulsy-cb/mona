# Trading Exit Optimizer

Système d'optimisation des sorties de trades avec deux approches:
1. **Heuristique** - Règles basées sur la volatilité et le momentum
2. **ML (PPO)** - Reinforcement Learning pour apprentissage adaptatif

## Installation

```bash
cd /home/azy/mona
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Backtest avec trailing stop standard
python -m src.backtest.runner --strategy original

# Backtest avec approche heuristique
python -m src.backtest.runner --strategy heuristic

# Entraînement ML
python -m src.ml.train

# Backtest avec modèle ML
python -m src.backtest.runner --strategy ml --model models/best_model.zip
```

## Structure

```
src/
├── core/           # Types, configs, utilitaires
├── data/           # Chargement et conversion données
├── indicators/     # Indicateurs techniques
├── backtest/       # Engine de simulation
├── strategy/       # Stratégies de sortie
├── ml/             # Environnement et entraînement RL
└── mt5/            # Connecteur MetaTrader 5
```
# mona
# Entraînement avec features pré-calculées (par défaut)
python -m src.ml.train --data XAUUSD.parquet --timesteps 100000

# Forcer le recalcul
python -m src.ml.train --force-recompute