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

## 🐛 Current Status (2025-11-25)

### Issue: 0 Trades at Checkpoint 250K ❌

**Symptoms**:
- Total Trades: 0
- Total Reward: +110,232 (positive but passive)
- Actions: SELL 0%, HOLD 0%, BUY 0%

**Hypotheses**:
1. ⚠️ **Reward scale dilutes everything** (0.3× if 0 trades)
2. 🔍 Demonstration Learning phase not detected
3. ⚠️ Over-Trading Protection too strict (blocks first 10 steps)
4. 🔍 Action Masking too aggressive
5. 🔍 RSI thresholds still too narrow

**Fixes Applied (Not Yet Tested)**:
- [x] Trading Action Rewards +5.0 (protected at END)
- [x] RSI thresholds widened (30/70 → 40/60)
- [x] Rewards boosted (×2.5)
- [ ] Reward scale = 1.0 in Phase 1 (pending)
- [ ] Over-Trading Protection fix (pending)

See [DIAGNOSTIC_URGENT.md](DIAGNOSTIC_URGENT.md) for detailed analysis.

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

**Last Updated**: 2025-11-25
**Version**: V2.7 NUCLEAR (7 fixes + 3 critical updates)
**Status**: 🔥 Active Development - Resolving 0 trades issue
