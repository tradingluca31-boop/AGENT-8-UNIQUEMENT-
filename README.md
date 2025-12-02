# 🤖 Agent 8 - RL Trading Gold (XAUUSD)

> **Reinforcement Learning Agent for Mean Reversion Trading on Gold (XAUUSD) - M15 Timeframe**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

---

## 📋 Overview

**Agent 8** is a sophisticated Reinforcement Learning trading agent specializing in **mean reversion strategies** on the Gold (XAUUSD) market using **M15 (15-minute) timeframe**. Built with institutional-grade standards, it leverages **PPO (Proximal Policy Optimization)** and incorporates hedge fund-level techniques including:

- ✅ **Demonstration Learning** (curriculum-based training)
- ✅ **Adaptive Entropy Scheduling** (0.40 → 0.20)
- ✅ **Protected Reward Shaping** (+5.0 trading action rewards)
- ✅ **Risk Management** (FTMO-compliant)
- ✅ **100+ Technical Features** (RSI, MACD, ADX, COT data, macro events)

---

## 🎯 Strategy

**Type**: Mean Reversion
**Timeframe**: M15 (15 minutes)
**Market**: Gold (XAUUSD)

**Core Logic**:
- **Entry**: Oversold (RSI < 40) → BUY, Overbought (RSI > 60) → SELL
- **Exit**: Fixed TP (4R ~1%) or Fixed SL (ATR-based)
- **Hold Time**: 15 minutes - 4 hours (quick reversions)

**Risk Management**:
- Max Drawdown: < 10% (FTMO compliant)
- Daily Loss Limit: < 5%
- Position Sizing: Kelly Criterion
- Over-Trading Protection: Max 1 trade per 2.5 hours

---

## 🚀 Features

### 1. Advanced RL Architecture
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Action Space**: Discrete(3) - [0=SELL, 1=HOLD, 2=BUY]
- **Network**: [512, 512] neurons
- **Training**: 500K steps validation, 1M+ production

### 2. Institutional-Grade Reward System
- **Trading Action Rewards**: +5.0 (open), +5.0 (profitable close), -1.0 (loss)
- **Bonuses**: ×20 amplified (direction prediction, profit taking, loss cutting)
- **HOLD Penalty**: Exponential (-18.0 max after 20 consecutive holds)
- **Protected Rewards**: Added AFTER scaling (not diluted)

### 3. Demonstration Learning (Revolutionary!)
- **Phase 1** (0-100K): Force 100% of smart trades (RSI signals) + MEGA rewards (+10.0)
- **Phase 2** (100K-300K): Reduce forcing (50% → 0%), rewards (+5.0)
- **Phase 3** (300K-500K): Full autonomy, amplified rewards (+2.0)

### 4. Anti-Mode Collapse Mechanisms
- **Adaptive Entropy**: 0.40 (high exploration) → 0.20 (moderate)
- **Action Masking**: Block if ≥5 times in last 10 actions
- **Forced Trading**: Safety net if 0 trades after 1000 steps
- **Diversity Rewards**: Shannon entropy-based

### 5. ALL Features (No Selection - Full Dataset)
**Technical Indicators**:
- Price Action: SMA, EMA, Bollinger Bands
- Momentum: RSI, MACD, ADX, Stochastic
- Volatility: ATR, Historical Volatility

**Correlations**:
- Gold vs EURUSD, USDJPY, DXY, USDCHF, AUDJPY, Silver

**Macro/Fundamental**:
- COT Data (CFTC): Noncomm/Comm positions, divergence
- US Macro Events: FOMC, NFP, CPI, PPI
- Seasonality: Strong months, best months (Seasonax)

---

## 📊 Performance Targets

**Validation (500K steps)**:
- Total Trades: > 100
- Action Diversity: ~30% SELL, ~30% HOLD, ~30% BUY
- Entropy: > 0.20

**Production (1M+ steps)**:
- ROI: 10-15% (test period 2022-2024)
- Sharpe Ratio: > 1.2
- Max Drawdown: < 8%
- Win Rate: > 50%
- Profit Factor: > 1.5

**FTMO Compliance**:
- Max Drawdown: < 10% ✅
- Daily Loss: < 5% ✅
- Minimum Trading Days: 4+ ✅

---

## 🛠️ Installation

### Requirements
```bash
Python 3.8+
NumPy >= 1.20
Pandas >= 1.3
Stable-Baselines3 >= 2.0
Gymnasium >= 0.28
TensorBoard (optional, for monitoring)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Requirements
- **XAUUSD M15**: 2008-2025 (~300,000 bars)
- **XAUUSD H1**: 2008-2025 (correlations)
- **XAUUSD D1**: 2008-2025 (trends)
- **Correlations**: EURUSD, USDJPY, DXY, etc. (same period)
- **COT Data**: Gold Futures weekly positions (CFTC)
- **US Macro Events**: FOMC, NFP, CPI (2008-2025)

---

## 🚀 Quick Start

### 1. Training (500K Validation - 40 min)
```bash
cd "AGENT 8 UNIQUEMENT"
python train.py
```

**Outputs**:
- `checkpoints/agent8_checkpoint_50000_steps.zip` (every 50K)
- `checkpoints/agent8_500k_final.zip` (final model)
- `checkpoints_analysis/checkpoint_*.csv` (stats)
- `training_summary_500k.json`

### 2. Interview Diagnostic (3 min)
```bash
python interview.py
```

Asks 8 critical questions to diagnose agent behavior:
1. Are fixes activated?
2. Is Demonstration Learning forcing trades?
3. Are Trading Action Rewards applied?
4. Why does agent choose HOLD?
5. Are logits balanced?
6. Is entropy correct?
7. What does agent see in observations?
8. What's the ROOT CAUSE?

**Output**: `DIAGNOSTIC_REPORT_V27_YYYYMMDD_HHMMSS.txt`

### 3. Backtest (Production Model)
```bash
python backtest.py --model checkpoints/agent8_500k_final.zip --start 2022-01-01 --end 2024-12-31
```

---

## 📂 Project Structure

```
AGENT 8 UNIQUEMENT/
├── trading_env.py              # Main environment (7 NUCLEAR fixes)
├── train.py                    # Training script (PPO 500K)
├── interview.py                # Diagnostic tool (8 questions)
├── backtest.py                 # Backtesting script
├── RUN_TRAINING.bat            # Training launcher (Windows)
├── RUN_INTERVIEW.bat           # Interview launcher (Windows)
├── checkpoints/                # Saved models
├── checkpoints_analysis/       # Training stats (CSV)
├── V2.7_CHANGES.md             # 7 NUCLEAR fixes documentation
├── V2.7_CRITICAL_FIXES_APPLIED.md  # Latest fixes (3 critical)
├── DIAGNOSTIC_URGENT.md        # Current issues & hypotheses
└── README.md                   # This file
```

---

## 🔬 Methodology

### Inspired by Hedge Funds Standards

**Renaissance Technologies**:
- Adaptive entropy scheduling
- Curriculum learning (easy → hard)
- Feature selection (SHAP analysis)

**Two Sigma**:
- Real-time monitoring (TensorBoard)
- Behavioral diversity enforcement (Shannon entropy)
- Anti-mode collapse mechanisms

**Citadel**:
- Constrained exploration (entropy limits)
- Risk management (Kelly Criterion, VaR)
- Demonstration-based learning

### Academic References

1. **Haarnoja et al. (2018)** - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
2. **Hester et al. (2018)** - "Deep Q-learning from Demonstrations"
3. **Narvekar et al. (2020)** - "Curriculum Learning for Reinforcement Learning Domains"
4. **Ng et al. (1999)** - "Policy Invariance Under Reward Transformations"

---

## 🐛 Bugs Résolus (2025-12-02)

### Bug 1 : Early Returns dans `_calculate_reward()` ✅ RÉSOLU

**Problème** : Les `return` statements bypassaient le reward +5.0 pour les trades.

**Solution** : Changé tous les `return X` en `reward += X` pour accumuler.

**Fichier** : `environment/trading_env.py`

---

### Bug 2 : prices_df utilisant mauvaises données ✅ RÉSOLU

**Problème** : `df_full` ne contient que des colonnes `_close` (EURUSD ~$1.5), pas de vraies données OHLCV Gold.

**Solution** : Utiliser `auxiliary_data['xauusd_raw']['H1']` pour les prix réels Gold (~$1000).

**Fichiers** :
- `analysis/interview_env_only.py`
- `analysis/quick_diagnostic.py`
- `training/train_smoke_test.py`

---

### Bug 3 : FIX 8 bloquant TOUS les trades ✅ RÉSOLU

**Problème** : `last_trade_open_step` n'était pas reset dans `reset()`, causant des différences négatives :
```
current_step: 44290
last_trade_open_step: 61667  (de l'épisode précédent!)
Différence: -17377 bars
-17377 < 10 = TRUE → TOUS LES TRADES BLOQUÉS
```

**Solution** : Ajouté dans `reset()` :
```python
self.last_trade_open_step = -10  # Allow first trade immediately
self.position_opened_this_step = False
self.position_closed_this_step = False
self.last_closed_pnl = 0.0
self.demonstration_trade_this_step = False
```

**Fichier** : `environment/trading_env.py` (lignes 398-407)

---

## 🔬 Scripts de Diagnostic

### 1. Interview Trades (`analysis/interview_trades.py`)

**Le plus complet** - Pose 5 questions critiques :

| Question | But |
|----------|-----|
| Q1 | Les positions S'OUVRENT-elles ? |
| Q2 | Le TP/SL est-il ATTEIGNABLE ? |
| Q3 | Que se passe-t-il sur 100 steps ? |
| Q4 | FIX 8 (Over-Trading) bloque-t-il ? |
| Q5 | Simulation longue (500 steps) |

```bash
python analysis/interview_trades.py
```
**Durée** : ~3 minutes

### 2. Interview Env Only (`analysis/interview_env_only.py`)

Test l'environnement **sans modèle** - Vérifie fixes V2.7, rewards, `_open_position()`.

### 3. Quick Diagnostic (`analysis/quick_diagnostic.py`)

10 actions BUY manuelles + test direct `_open_position()`.

### 4. Check Prices (`analysis/check_prices.py`)

Vérifie que les prix sont bien Gold (~$800-2000) et pas EURUSD (~$1.5).

### 5. Behavioral Analysis (`analysis/behavioral_analysis.py`) 🆕

**Psychanalyse de l'agent** - 8 questions comportementales :

| Question | Analyse |
|----------|---------|
| Q1 | Distribution des actions (BUY/SELL/HOLD %) |
| Q2 | Peur du risque? (préfère HOLD vs trade) |
| Q3 | Ferme-t-il ses positions? |
| Q4 | Rewards reçus (positifs/négatifs) |
| Q5 | Contexte marché quand il trade |
| Q6 | Ce qu'il "voit" (observations) |
| Q7 | Évolution sur 1000 steps |
| Q8 | Synthèse + hypothèses |

```bash
python analysis/behavioral_analysis.py
```
**Durée** : ~5 minutes
**Output** : Diagnostic comportemental avec profil (PASSIF/ACTIF/BLOQUÉ) et recommandations

### 6. Check Q-Values / Policy (`analysis/check_qvalues.py`) 🆕

**Regarde ce que l'agent "pense"** - Probabilités d'action du policy network :

```bash
python analysis/check_qvalues.py
python analysis/check_qvalues.py --model models/best_model.zip
python analysis/check_qvalues.py --n_samples 100
```

**Diagnostics**:
- Si P(HOLD) >> P(BUY) et P(SELL) → L'agent a appris que ne rien faire est "safe"
- Calcul de l'entropy pour vérifier exploration
- Mode collapse detection

### 7. Check Reward Function (`analysis/check_reward_function.py`) 🆕

**Checklist reward function** - 4 questions critiques :

| Question | Check |
|----------|-------|
| A | HOLD donne-t-il un reward (même petit)? |
| B | Trades perdants TROP pénalisés? |
| C | Reward UNIQUEMENT à la clôture? |
| D | Risk/Reward encouragé? |

```bash
python analysis/check_reward_function.py
```

**PIÈGE CLASSIQUE détecté** : Si reward qu'à clôture + pertes pénalisées → Agent apprend "ne jamais ouvrir = jamais de perte = SAFE"

### 8. Full Diagnostic (`analysis/full_diagnostic.py`) 🆕

**Diagnostic COMPLET** - 6 tests en un script :

| Test | Description |
|------|-------------|
| 1 | Distribution des actions (HOLD/BUY/SELL %) |
| 2 | Random vs Trained comparison |
| 3 | Observations normalization check |
| 4 | Reward per action type |
| 5 | Exploration (entropy) analysis |
| 6 | Positions analysis |

```bash
python analysis/full_diagnostic.py
python analysis/full_diagnostic.py --model models/best_model.zip --episodes 20
```

**Output** : Synthèse avec problèmes détectés + solutions recommandées

---

## 🔄 Workflow de Diagnostic

```
1. Problème détecté (0 trades)
         ↓
2. Lancer interview_trades.py
         ↓
3. Identifier la question qui échoue (Q1-Q5)
         ↓
4. Analyser le code source correspondant
         ↓
5. Appliquer le fix
         ↓
6. Relancer interview pour vérifier
         ↓
7. Lancer smoke test (10K steps)
         ↓
8. Si SUCCESS → Training complet (500K+ steps)
```

---

## 📦 Repositories

| Repo | Description |
|------|-------------|
| [AGENT-8-UNIQUEMENT-](https://github.com/tradingluca31-boop/AGENT-8-UNIQUEMENT-) | Code Agent 8 centralisé |
| [AMELIORATION-AGENT-SCRIPT](https://github.com/tradingluca31-boop/AMELIORATION-AGENT-SCRIPT) | Scripts d'amélioration & diagnostic |

---

## 🤝 Contributing

This is a personal research project. Contributions, suggestions, and discussions are welcome!

**Areas for Improvement**:
1. Resolve 0 trades issue (reward scale, demonstration learning)
2. Hyperparameter tuning (Optuna, Ray Tune)
3. Multi-timeframe fusion (M15 + H1 + D1)
4. Ensemble learning (combine with other agents)
5. Walk-forward optimization
6. Live trading integration (MetaTrader 5 API)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, feedback, or collaboration:
- **GitHub Issues**: [Report a bug or request a feature]
- **Email**: [Your email if you want]

---

## 🙏 Acknowledgments

- **Stable-Baselines3** team for excellent RL library
- **OpenAI Gym/Gymnasium** for standardized environments
- **CFTC** for COT data
- **Seasonax** for seasonality data
- **MetaQuotes** for MetaTrader 5 platform

---

## ⚠️ Disclaimer

**THIS IS RESEARCH SOFTWARE - NOT FINANCIAL ADVICE**

Trading financial instruments involves substantial risk of loss. Past performance is not indicative of future results. This agent is **experimental** and should **NOT** be used with real money without extensive testing, validation, and understanding of the risks involved.

**Use at your own risk. The authors assume no responsibility for financial losses.**

---

---

## 📜 Changelog

### [2025-12-02] - Session 3
- ✅ **ADD** `analysis/normalization_audit.py` - Audit Wall Street grade (47 features à normaliser)
- ✅ **ADD** 7 MEMORY features dans `trading_env.py` (like Agent 7)
  - win_rate, streak, avg_pnl, best, worst, win_count, loss_count
- ✅ **ADD** `analysis/normalization_audit_summary.json` - Résultats audit
- ✅ **FIX** `normalize_features_wallstreet()` - Normalisation automatique dans trading_env.py
  - PRICE RAW: Min-Max [0, 1]
  - RSI/Stoch: Divide by 100
  - VOLUME: Log + Z-score [-3, 3]
  - ATR: Percentile rank [0, 1]
  - MACD/ADX: Z-score [-3, 3]
- 🔍 **DIAGNOSTIC** 31 features CRITIQUES identifiées (prix raw 0-2000)
- 📊 **Observation space**: 209 base + 20 RL = **229 features** total

### [2025-12-02] - Session 2
- ✅ **ADD** `analysis/behavioral_analysis.py` - Psychanalyse de l'agent (8 questions)
- ✅ **ADD** `analysis/check_qvalues.py` - Vérification Q-values/Policy output
- ✅ **ADD** `analysis/check_reward_function.py` - Checklist reward function (4 questions)
- ✅ **ADD** `analysis/full_diagnostic.py` - Diagnostic COMPLET (6 tests)
- ✅ **FIX** Bug 3 - FIX 8 reset (`last_trade_open_step` non reset dans `reset()`)
- ✅ **VERIFIED** Interview trades: 50 positions ouvertes, 48 fermées

### [2025-12-02] - Session 1
- ✅ **FIX** Bug 1 - Early returns dans `_calculate_reward()` bypassaient rewards +5.0
- ✅ **FIX** Bug 2 - `prices_df` utilisait EURUSD (~$1.5) au lieu de Gold (~$2000)
- ✅ **ADD** `analysis/interview_trades.py` - Diagnostic 5 questions
- ✅ **ADD** `analysis/interview_env_only.py` - Test environnement sans modèle
- ✅ **ADD** `analysis/quick_diagnostic.py` - Test rapide 10 actions
- ✅ **ADD** `analysis/check_prices.py` - Vérification prix Gold vs EURUSD

### [2025-12-01] - Initial Setup
- 🚀 **INIT** Structure projet Agent 8
- 📦 **ADD** `environment/trading_env.py` - GoldTradingEnvAgent8 V2.7
- 📦 **ADD** `training/train_smoke_test.py` - Smoke test 10K steps

---

**Last Updated**: 2025-12-02
**Version**: V2.9 NORMALIZED (229 features - Wall Street grade normalization applied)
**Status**: ✅ Ready for training - All features normalized automatically
