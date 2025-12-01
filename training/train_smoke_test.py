r"""
SMOKE TEST - AGENT 8 (10K steps - 5 minutes)
================================================================================
Quick test to verify the agent TRADES with the new fixes.

SUCCESS CRITERIA:
  - Total trades > 5
  - At least 2 actions used (not 100% HOLD)

FIXES BEING TESTED:
  - Trading Action Rewards: +5.0 (protected at END)
  - RSI thresholds: 40/60 (widened)
  - Demonstration Learning: Phase 1 forces smart trades

Duration: ~5 minutes
================================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Add paths
training_dir = Path(__file__).resolve().parent
agent8_root = training_dir.parent
env_dir = agent8_root / "environment"

# GoldRL paths (where data_loader, feature_engineering, config are)
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

sys.path.insert(0, str(env_dir))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(goldrl_root))

print("="*80)
print("AGENT 8 - SMOKE TEST (10K steps)")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Working Directory: {agent8_root}")
print("="*80)
print()

# ================================================================================
# IMPORTS
# ================================================================================

print("[STEP 1/6] Importing modules...")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    print("           [OK] SB3 imported")
except ImportError as e:
    print(f"           [ERROR] SB3 import failed: {e}")
    sys.exit(1)

try:
    from trading_env import GoldTradingEnvAgent8
    print("           [OK] Environment imported")
except ImportError as e:
    print(f"           [ERROR] Environment import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("           [OK] Data modules imported")
except ImportError as e:
    print(f"           [ERROR] Data modules import failed: {e}")
    sys.exit(1)

print()

# ================================================================================
# HYPERPARAMETERS (SMOKE TEST)
# ================================================================================

HYPERPARAMS = {
    'total_timesteps': 10_000,  # SMOKE TEST: Only 10K steps!
    'algorithm': 'PPO',

    # Data splits
    'train_start': '2008-01-01',
    'train_end': '2020-12-31',

    # PPO config
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.30,  # Fixed entropy for smoke test (high exploration)
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,

    # Network
    'policy_kwargs': {
        'net_arch': [512, 512]
    },
}

# ================================================================================
# LOAD DATA
# ================================================================================

print("[STEP 2/6] Loading data...")
start_load = time.time()

df_full, auxiliary_data = load_data_agent8()

# Filter to training period
df_train = df_full.loc[HYPERPARAMS['train_start']:HYPERPARAMS['train_end']]

# Filter auxiliary data
for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        mask = (auxiliary_data['xauusd_raw'][tf].index >= HYPERPARAMS['train_start']) & \
               (auxiliary_data['xauusd_raw'][tf].index <= HYPERPARAMS['train_end'])
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc[mask]

# Extract OHLCV
ohlcv_cols = [col for col in df_train.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
prices_df = df_train[ohlcv_cols[:5]].copy()
prices_df.columns = ['open', 'high', 'low', 'close', 'volume']

# Calculate features
features_df = calculate_all_features(df_train, auxiliary_data)

load_time = time.time() - start_load
print(f"           [OK] Data loaded in {load_time:.1f}s")
print(f"           Training period: {features_df.index[0]} to {features_df.index[-1]}")
print(f"           Total bars: {len(features_df):,}")
print(f"           Features: {features_df.shape[1]}")
print()

# ================================================================================
# CREATE ENVIRONMENT
# ================================================================================

print("[STEP 3/6] Creating environment...")

env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_position_size=1.0,
    transaction_cost=0.0001,
    verbose=False,
    global_timestep=0
)

print(f"           [OK] Environment created")
print(f"           Action space: {env.action_space}")
print(f"           Observation space: {env.observation_space.shape}")
print()

# ================================================================================
# TRACKING CALLBACK
# ================================================================================

class SmokeTestCallback(BaseCallback):
    """Track actions and trades during smoke test"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY
        self.total_trades = 0
        self.last_print = 0

    def _on_step(self) -> bool:
        # Get action
        action = self.locals.get('actions', [1])[0]
        if isinstance(action, np.ndarray):
            action = int(action[0])

        # Track action
        if action in self.action_counts:
            self.action_counts[action] += 1

        # Print progress every 2000 steps
        if self.num_timesteps - self.last_print >= 2000:
            total = sum(self.action_counts.values())
            if total > 0:
                pct_sell = self.action_counts[0] / total * 100
                pct_hold = self.action_counts[1] / total * 100
                pct_buy = self.action_counts[2] / total * 100

                # Get trades from env
                if hasattr(self.training_env.envs[0], 'trades'):
                    self.total_trades = len(self.training_env.envs[0].trades)

                print(f"  Step {self.num_timesteps:,}: SELL {pct_sell:.1f}% | HOLD {pct_hold:.1f}% | BUY {pct_buy:.1f}% | Trades: {self.total_trades}")

            self.last_print = self.num_timesteps

        return True

    def get_results(self):
        total = sum(self.action_counts.values())
        if total == 0:
            return None

        return {
            'total_steps': total,
            'sell_pct': self.action_counts[0] / total * 100,
            'hold_pct': self.action_counts[1] / total * 100,
            'buy_pct': self.action_counts[2] / total * 100,
            'total_trades': self.total_trades
        }

# ================================================================================
# CREATE MODEL
# ================================================================================

print("[STEP 4/6] Creating PPO model...")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=HYPERPARAMS['learning_rate'],
    n_steps=HYPERPARAMS['n_steps'],
    batch_size=HYPERPARAMS['batch_size'],
    n_epochs=HYPERPARAMS['n_epochs'],
    gamma=HYPERPARAMS['gamma'],
    gae_lambda=HYPERPARAMS['gae_lambda'],
    clip_range=HYPERPARAMS['clip_range'],
    ent_coef=HYPERPARAMS['ent_coef'],
    vf_coef=HYPERPARAMS['vf_coef'],
    max_grad_norm=HYPERPARAMS['max_grad_norm'],
    policy_kwargs=HYPERPARAMS['policy_kwargs'],
    verbose=0,
    device='auto'
)

print(f"           [OK] Model created")
print(f"           Device: {model.device}")
print()

# ================================================================================
# TRAIN (SMOKE TEST)
# ================================================================================

print("[STEP 5/6] Training (10K steps)...")
print("-" * 60)

callback = SmokeTestCallback(verbose=1)
start_train = time.time()

model.learn(
    total_timesteps=HYPERPARAMS['total_timesteps'],
    callback=callback,
    progress_bar=False
)

train_time = time.time() - start_train
print("-" * 60)
print(f"           [OK] Training completed in {train_time:.1f}s")
print()

# ================================================================================
# RESULTS
# ================================================================================

print("[STEP 6/6] Results...")
print("=" * 60)

results = callback.get_results()

if results:
    print(f"  Total Steps:  {results['total_steps']:,}")
    print(f"  SELL:         {results['sell_pct']:.1f}%")
    print(f"  HOLD:         {results['hold_pct']:.1f}%")
    print(f"  BUY:          {results['buy_pct']:.1f}%")
    print(f"  Total Trades: {results['total_trades']}")
    print()

    # SUCCESS/FAILURE
    print("=" * 60)
    if results['total_trades'] >= 5:
        print("✅ SUCCESS! Agent is TRADING!")
        print(f"   {results['total_trades']} trades detected")
        print("   → Ready for full training (500K steps)")
    elif results['total_trades'] > 0:
        print("⚠️ PARTIAL SUCCESS - Agent trades but not enough")
        print(f"   Only {results['total_trades']} trades (expected >5)")
        print("   → May need more steps or stronger fixes")
    else:
        print("❌ FAILURE - Agent still NOT TRADING!")
        print("   0 trades detected")
        print("   → Check docs/DIAGNOSTIC_URGENT.md for fixes")
    print("=" * 60)

    # Save results
    output_dir = agent8_root / "outputs" / "checkpoints_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_timesteps': HYPERPARAMS['total_timesteps'],
            'results': results,
            'success': results['total_trades'] >= 5,
            'train_time_seconds': train_time
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
else:
    print("❌ ERROR - No results collected!")

print()
print("=" * 80)
print(f"SMOKE TEST COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
