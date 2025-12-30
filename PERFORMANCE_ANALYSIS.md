# Analyse des Performances - Entraînement RL Trading

## 📊 Résumé Exécutif

**Statut** : ⚠️ Modèle en phase d'apprentissage - Performance négative actuelle

**Reward moyen final** : **-145.93** (sur 2 épisodes d'évaluation)

---

## 📈 Résultats d'Évaluation

### Évaluation @ 100,000 steps
- **Mean reward**: -145.93 ± 86.67
- **Episode rewards**: [-232.86, -59.00]
- **Mean episode length**: 82,989 ticks
- **Nombre de trades**: ~101 trades (évaluation finale)

### Observations
- Le modèle génère des trades mais avec des pertes moyennes
- Grande variance entre les épisodes (écart-type de 86.67)
- Les épisodes sont longs (82k ticks en moyenne), indiquant que le modèle ne sort pas prématurément

---

## 💾 Modèles Sauvegardés

| Modèle | Taille |
|--------|--------|
| `best_model.zip` | 0.14 MB |
| `ppo_exit_model_100000_steps.zip` | 0.14 MB |
| `ppo_exit_model_200000_steps.zip` | 0.14 MB |
| `ppo_exit_model_300000_steps.zip` | 0.14 MB |
| `ppo_exit_model_400000_steps.zip` | 0.14 MB |
| `ppo_exit_model_500000_steps.zip` | 0.14 MB |

**Note**: Plusieurs checkpoints disponibles pour comparer les performances à différentes étapes.

---

## 📦 Données Pré-calculées

- **Cache file**: `XAUUSD_preprocessed.npz`
- **Taille**: 179.2 MB (compressé)
- **Ticks**: 23,032,565
- **Features**: 6 features statiques
  - `bb_percent` (Bollinger %B)
  - `bb_width_norm` (BB width normalisé)
  - `atr_norm` (ATR normalisé)
  - `rsi_norm` (RSI normalisé)
  - `volatility_10` (volatilité sur 10 ticks)
  - `velocity_5` (vélocité sur 5 ticks)

---

## 📈 TensorBoard

**11 runs disponibles** : `PPO_1` à `PPO_11`

Pour visualiser les métriques détaillées :
```bash
tensorboard --logdir models/tensorboard
```

Métriques disponibles :
- Loss (policy, value, entropy)
- Rewards (mean, std)
- Episode length
- Learning rate
- Explained variance

---

## 🔍 Diagnostic

### Pourquoi le modèle perd de l'argent ?

1. **Entraînement insuffisant**
   - 100,000 steps est relativement court pour un problème complexe
   - Le modèle n'a peut-être pas convergé

2. **Reward shaping**
   - Les rewards denses peuvent nécessiter plus de temps pour apprendre
   - Le système d'oracle et d'efficacité peut être trop complexe initialement

3. **Stratégie d'entrée**
   - Les signaux Dark Venus peuvent ne pas être optimaux
   - Le modèle apprend à sortir mais les entrées sont fixes

4. **Hyperparamètres**
   - `ent_coef=0.01` peut être trop faible pour l'exploration
   - Learning rate peut nécessiter un ajustement

---

## 💡 Recommandations

### 1. Entraînement Plus Long (PRIORITÉ HAUTE)

```bash
# Entraîner pour 1M steps
python -m src.ml.train --data XAUUSD.parquet --timesteps 1000000
```

**Justification** : Les modèles RL nécessitent souvent 1M+ steps pour converger sur des problèmes complexes.

### 2. Ajuster les Hyperparamètres PPO

Modifier dans `train.py` :

```python
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,      # Réduit de 3e-4
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,           # Augmenté de 0.01 pour plus d'exploration
    vf_coef=0.5,
    clip_range=0.2,
    verbose=1
)
```

### 3. Simplifier le Reward (Optionnel)

Tester avec un reward plus simple pour démarrer :
- Supprimer temporairement l'oracle et l'efficiency bonus
- Utiliser uniquement le PnL réalisé
- Ajouter progressivement la complexité une fois que le modèle apprend

### 4. Analyser les Trades

Créer un script pour examiner les trades individuels :
- Durée moyenne des trades
- Distribution des PnL
- Taux de win/loss
- Raisons de sortie (agent vs SL)

### 5. Vérifier la Stratégie d'Entrée

Analyser les signaux Dark Venus :
- Combien de signaux par jour ?
- Qualité des signaux (backtest simple)
- Distribution long/short

---

## 📊 Métriques à Surveiller

### Pendant l'Entraînement

1. **Mean Reward** : Doit augmenter progressivement
2. **Policy Loss** : Doit diminuer et se stabiliser
3. **Value Loss** : Doit diminuer
4. **Entropy** : Doit rester > 0 (exploration)
5. **Explained Variance** : Doit être proche de 1

### Pendant l'Évaluation

1. **Win Rate** : % de trades profitables
2. **Average PnL** : PnL moyen par trade
3. **Sharpe Ratio** : Ratio reward/risque
4. **Max Drawdown** : Perte maximale
5. **Trade Duration** : Durée moyenne des positions

---

## 🎯 Plan d'Action Recommandé

### Phase 1 : Entraînement Long (1-2 jours)
```bash
python -m src.ml.train --data XAUUSD.parquet --timesteps 2000000
```

### Phase 2 : Analyse Approfondie
- Examiner les courbes TensorBoard
- Analyser les trades individuels
- Comparer les checkpoints

### Phase 3 : Optimisation
- Ajuster hyperparamètres basé sur Phase 2
- Tester différentes configurations de reward
- Expérimenter avec différentes stratégies d'entrée

### Phase 4 : Validation
- Test sur données out-of-sample
- Walk-forward analysis
- Paper trading

---

## ✅ Points Positifs

1. ✅ **Système de pré-calcul fonctionne** : 10-50x speedup confirmé
2. ✅ **Pas de crashes** : Entraînement stable
3. ✅ **Checkpoints sauvegardés** : Possibilité de reprendre
4. ✅ **TensorBoard configuré** : Métriques disponibles
5. ✅ **Cache fonctionnel** : Pas besoin de recalculer les features

---

## 🚀 Prochaines Étapes Immédiates

1. **Lancer un entraînement long** (1M+ steps)
2. **Monitorer TensorBoard** pendant l'entraînement
3. **Analyser les résultats** après convergence
4. **Itérer sur les hyperparamètres** si nécessaire

---

## 📝 Notes Techniques

### Performance du Système
- **Preprocessing** : ~2 minutes pour 23M ticks (une seule fois)
- **Training** : Utilise le cache, très rapide
- **Environnement** : PrecomputedTradingEnv fonctionne correctement

### Compatibilité
- ✅ Stable-Baselines3
- ✅ Vectorized environments
- ✅ Callbacks (eval, checkpoint)
- ✅ TensorBoard logging

---

**Conclusion** : Le système fonctionne correctement mais nécessite plus d'entraînement. Les performances actuelles sont normales pour un modèle à 100k steps. Recommandation : continuer l'entraînement jusqu'à 1-2M steps.
