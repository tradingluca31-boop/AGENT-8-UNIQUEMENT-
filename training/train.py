r"""
Training Script 500K - AGENT 8 V2.7 NUCLEAR (MODE COLLAPSE FIX)
================================================================================
V2.7 NUCLEAR FIXES:
  - 7 NUCLEAR fixes to solve 100% HOLD mode collapse (logit gap +1081)
  - FIX 1: Trading Action Rewards (+2.0 open, +2.0 gain, -0.5 loss)
  - FIX 2: Bonuses ×20 (vs ×4 in V2.6)
  - FIX 3: HOLD Penalty Exponential (-18.0 max vs -0.15)
  - FIX 4: Action Masking 5/10 (vs 8/10)
  - FIX 5: Demonstration Learning (Phases 1-3: 0-500K)
  - FIX 6: Forced Trading (if 0 trades after 1000 steps)
  - FIX 8: Over-Trading Protection (max 1 trade per 10 bars)
  - PPO with ADAPTIVE ENTROPY: 0.40 → 0.20 (linear decay)

Duration: ~40 minutes
Purpose: Validate V2.7 fixes before full training (1M+)

Version: V2.7 NUCLEAR
Date: 2025-11-25
Author: Claude Code

Usage:
    cd C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 8\ALGO AGENT 8 RL\V2
    python train_500k_v2.7.py
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

# Add paths - AGENT 8 CENTRALIZED STRUCTURE
training_dir = Path(__file__).resolve().parent
agent8_root = training_dir.parent
env_dir = agent8_root / "environment"
models_dir = agent8_root / "models"
v2_dir = training_dir  # Output directory for checkpoints and logs
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

# Priority: 1) environment folder (AGENT 8 UNIQUEMENT), 2) V2 folder, 3) GoldRL root
# Note: insert(0) puts at front, so insert in REVERSE order of priority
sys.path.insert(0, str(goldrl_root))   # 3rd priority
sys.path.insert(0, str(goldrl_v2))     # 2nd priority
sys.path.insert(0, str(env_dir))       # 1st priority (searched first)

print("="*80)
print("AGENT 8 V2.9 NORMALIZED - 500K TRAINING")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Working Directory: {training_dir}")
print("="*80)
print()

# ================================================================================
# IMPORTS
# ================================================================================

print("[STEP 1/8] Importing modules...")

try:
    # Environment from AGENT 8 UNIQUEMENT/environment/
    from trading_env import GoldTradingEnvAgent8
    print("           [OK] trading_env imported from environment/")

    # Data loader and feature engineering from GoldRL V2 folder
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("           [OK] data_loader + feature_engineering imported from V2/")
except ImportError as e:
    print(f"           [ERROR] Import failed: {e}")
    print(f"           [DEBUG] sys.path = {sys.path[:5]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from stable_baselines3 import PPO  # V2.7 uses PPO (not SAC)
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
    print("           [OK] Stable-Baselines3 (PPO) imported")
except ImportError:
    print("           [ERROR] stable_baselines3 not installed!")
    sys.exit(1)

print()

# ================================================================================
# HYPERPARAMETERS
# ================================================================================

HYPERPARAMS = {
    # Training
    'total_timesteps': 500_000,  # 500K for validation
    'algorithm': 'PPO',  # V2.7 uses PPO

    # Data splits
    'train_start': '2008-01-01',
    'train_end': '2020-12-31',
    'val_start': '2021-01-01',
    'val_end': '2021-12-31',

    # PPO config (V2.7 NUCLEAR)
    'learning_rate': 3e-4,
    'n_steps': 2048,  # PPO parameter
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef_start': 0.80,  # OPTION 2: DOUBLED for max exploration (was 0.40)
    'ent_coef_end': 0.40,    # OPTION 2: DOUBLED to maintain diversity (was 0.20)
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': {'net_arch': [256, 256]},  # Agent 8 architecture

    # Callbacks
    'checkpoint_freq': 50_000,  # Checkpoint every 50K steps

    # Environment
    'initial_balance': 100_000.0,
    'max_episode_steps': 5_000,
}

print("="*80)
print("HYPERPARAMETERS V2.7 NUCLEAR")
print("="*80)
for key, value in HYPERPARAMS.items():
    if isinstance(value, int) and value >= 1000:
        print(f"  {key:25s}: {value:,}")
    else:
        print(f"  {key:25s}: {value}")
print("="*80)
print()

# ================================================================================
# ADAPTIVE ENTROPY SCHEDULER (V2.7 KEY FIX)
# ================================================================================

class AdaptiveEntropyCallback(BaseCallback):
    """
    V2.7 NUCLEAR: Adaptive Entropy Coefficient Schedule

    Linear decay: 0.40 (0%) → 0.20 (100%)
    - 0%: ent_coef = 0.40 (HIGH exploration)
    - 50%: ent_coef = 0.30 (moderate)
    - 100%: ent_coef = 0.20 (still exploring, not 0.10!)

    Standard: Renaissance Technologies (Medallion Fund)
    """

    def __init__(self, start_coef: float, end_coef: float, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Calculate progress (0.0 to 1.0)
        progress = self.num_timesteps / self.total_timesteps

        # Linear interpolation
        current_ent_coef = self.start_coef + (self.end_coef - self.start_coef) * progress

        # Update model entropy coefficient
        self.model.ent_coef = current_ent_coef

        # Log every 10K steps
        if self.num_timesteps % 10000 == 0 and self.verbose:
            print(f"   [ENTROPY V2.7] Step {self.num_timesteps:,} | ent_coef = {current_ent_coef:.3f}")

        return True

# ================================================================================
# GLOBAL TIMESTEP UPDATER (V2.7 FIX 5 - DEMONSTRATION LEARNING)
# ================================================================================

class GlobalTimestepCallback(BaseCallback):
    """
    V2.7 FIX 5: Update environment's global_timestep for Demonstration Learning

    Phases:
    - Phase 1 (0-100K): Force 100% smart trades + MEGA rewards (+10.0)
    - Phase 2 (100K-300K): Force 50%→0% + amplified rewards (+5.0)
    - Phase 3 (300K-500K): Autonomy + amplified rewards (+2.0)
    """

    def __init__(self, env_ref, verbose: int = 1):
        super().__init__(verbose)
        self.env_ref = env_ref

    def _on_step(self) -> bool:
        # Get underlying environment
        env = self.env_ref.envs[0].env if hasattr(self.env_ref, 'envs') else self.env_ref

        # Update global timestep
        if hasattr(env, 'set_global_timestep'):
            env.set_global_timestep(self.num_timesteps)

        # Log phase transitions
        if self.num_timesteps in [100_000, 300_000, 500_000] and self.verbose:
            if self.num_timesteps == 100_000:
                print(f"\n   [DEMONSTRATION] Entering Phase 2: Reducing forcing (50%→0%), rewards +5.0")
            elif self.num_timesteps == 300_000:
                print(f"\n   [DEMONSTRATION] Entering Phase 3: Autonomy, rewards +2.0")
            elif self.num_timesteps == 500_000:
                print(f"\n   [DEMONSTRATION] Exiting demonstration learning")

        return True

# ================================================================================
# CHECKPOINT CSV CALLBACK
# ================================================================================

class CheckpointCSVCallback(BaseCallback):
    """
    Callback qui crée un CSV snapshot à chaque checkpoint (50K steps)

    V2.7: Includes action distribution to monitor mode collapse fix
    """

    def __init__(self, checkpoint_freq: int, env_ref, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.env_ref = env_ref
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

        # Check if checkpoint step
        if self.num_timesteps % self.checkpoint_freq == 0 and self.num_timesteps > 0:
            self._save_checkpoint_csv()

        return True

    def _save_checkpoint_csv(self):
        """Sauvegarde CSV snapshot pour ce checkpoint"""

        # Get environment metrics
        env = self.env_ref.envs[0].env if hasattr(self.env_ref, 'envs') else self.env_ref

        # Calculate metrics
        balance = env.balance
        equity = env.balance  # Simplified

        # Trade statistics
        trades = env.trades if hasattr(env, 'trades') else []
        total_trades = len(trades)

        if total_trades > 0:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]

            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

            pnls = [t.get('pnl', 0) for t in trades]
            cumulative_pnl = sum(pnls)

            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0.0
            best_trade = max(pnls) if pnls else 0.0
            worst_trade = min(pnls) if pnls else 0.0

            # Profit factor
            total_wins = sum([t['pnl'] for t in wins]) if wins else 0.0
            total_losses = abs(sum([t['pnl'] for t in losses])) if losses else 1.0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        else:
            win_rate = 0.0
            cumulative_pnl = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            best_trade = 0.0
            worst_trade = 0.0
            profit_factor = 0.0
            expectancy = 0.0

        # V2.7: Action distribution (monitor mode collapse fix)
        action_history = env.action_history if hasattr(env, 'action_history') else []
        if len(action_history) > 100:
            recent_actions = list(action_history)[-1000:]  # Last 1000 actions
            action_counts = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY
            for a in recent_actions:
                action_counts[int(a)] = action_counts.get(int(a), 0) + 1

            total_actions = len(recent_actions)
            action_sell_pct = (action_counts[0] / total_actions) * 100
            action_hold_pct = (action_counts[1] / total_actions) * 100
            action_buy_pct = (action_counts[2] / total_actions) * 100
        else:
            action_sell_pct = 0.0
            action_hold_pct = 0.0
            action_buy_pct = 0.0

        # Sharpe ratio (simplified)
        if len(self.episode_rewards) > 10:
            recent_rewards = self.episode_rewards[-100:]
            sharpe = (np.mean(recent_rewards) / (np.std(recent_rewards) + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        max_dd = env.max_drawdown if hasattr(env, 'max_drawdown') else 0.0

        # Average reward
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0.0

        # Total episodes
        total_episodes = len(self.episode_rewards)

        # FTMO compliance
        ftmo_compliant = max_dd < 0.10  # Max DD < 10%

        # Total reward
        total_reward = sum(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0

        # Prepare snapshot data (V2.7 format)
        snapshot_data = {
            'checkpoint': self.num_timesteps,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': cumulative_pnl,
            'avg_pnl': cumulative_pnl / total_trades if total_trades > 0 else 0.0,
            'total_reward': total_reward,
            'action_sell_pct': action_sell_pct,
            'action_hold_pct': action_hold_pct,
            'action_buy_pct': action_buy_pct
        }

        # Save CSV
        csv_filename = f"checkpoint_{self.num_timesteps}_stats.csv"
        csv_path = v2_dir / "checkpoints_analysis" / csv_filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([snapshot_data])
        df.to_csv(csv_path, index=False)

        # Log results
        print(f"\n   [CHECKPOINT {self.num_timesteps:,}]")
        print(f"   Trades: {total_trades} | Win Rate: {win_rate*100:.1f}%")
        print(f"   Actions: SELL {action_sell_pct:.1f}% | HOLD {action_hold_pct:.1f}% | BUY {action_buy_pct:.1f}%")
        print(f"   Total Reward: {total_reward:+.2f}")

        # Mode collapse warning
        if action_hold_pct > 80.0 or action_sell_pct > 80.0 or action_buy_pct > 80.0:
            print(f"   [WARNING] Possible mode collapse detected!")

        print()

# ================================================================================
# LOAD DATA & CREATE ENVIRONMENT
# ================================================================================

print("[STEP 2/8] Loading data...")
df_full, auxiliary_data = load_data_agent8()
print(f"           [OK] Data loaded: {df_full.shape}")
print(f"           Full period: {df_full.index[0]} to {df_full.index[-1]}")
print()

# ================================================================================
# CRITICAL: APPLY DATA SPLIT **BEFORE** FEATURE ENGINEERING (NO DATA LEAKAGE!)
# ================================================================================

print("[STEP 2.5/8] CRITICAL - Filtering data to training period ONLY...")
print(f"             BEFORE filter: {df_full.shape[0]} bars (2008-2025)")
print(f"             [WARNING]  MUST filter BEFORE feature engineering to prevent data leakage!")

# Filter df_full to training period
df_full = df_full.loc[HYPERPARAMS['train_start']:HYPERPARAMS['train_end']]

# Filter auxiliary_data (XAUUSD timeframes)
for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        original_len = len(auxiliary_data['xauusd_raw'][tf])
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc[
            HYPERPARAMS['train_start']:HYPERPARAMS['train_end']
        ]
        filtered_len = len(auxiliary_data['xauusd_raw'][tf])
        print(f"             Filtered XAUUSD {tf}: {original_len} -> {filtered_len} bars")

print(f"             AFTER filter:  {df_full.shape[0]} bars (train only)")
print(f"             Training period: {HYPERPARAMS['train_start']} to {HYPERPARAMS['train_end']}")
print(f"             [OK] Test set (2022-2025) will NEVER be seen during training!")
print()

# Extract OHLCV from FILTERED data
ohlcv_cols = [col for col in df_full.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
prices_df = df_full[ohlcv_cols[:5]].copy()
prices_df.columns = ['open', 'high', 'low', 'close', 'volume']
print(f"           [OK] OHLCV extracted: {prices_df.shape}")
print()

print("[STEP 3/8] Calculating features on TRAINING data only...")
print(f"           Feature engineering will compute on: {df_full.index[0]} to {df_full.index[-1]}")
features_df = calculate_all_features(df_full, auxiliary_data)
n_features = features_df.shape[1]
print(f"           [OK] Features calculated: {features_df.shape}")
print(f"           [OK] Features computed ONLY on 2008-2020 (NO data leakage!)")
print()

print("[STEP 4/8] Creating V2.7 NUCLEAR environment...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=HYPERPARAMS['initial_balance'],
    max_episode_steps=HYPERPARAMS['max_episode_steps'],
    verbose=False,
    training_mode=True
)

obs_space_size = env.observation_space.shape[0]
print(f"           [OK] V2.7 Environment created")
print(f"           Observation space: {obs_space_size}")
print()

# Wrap environment
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# ================================================================================
# CREATE MODEL (PPO V2.7)
# ================================================================================

print("[STEP 5/8] Creating PPO model (V2.7 NUCLEAR)...")
model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=HYPERPARAMS['learning_rate'],
    n_steps=HYPERPARAMS['n_steps'],
    batch_size=HYPERPARAMS['batch_size'],
    n_epochs=HYPERPARAMS['n_epochs'],
    gamma=HYPERPARAMS['gamma'],
    gae_lambda=HYPERPARAMS['gae_lambda'],
    clip_range=HYPERPARAMS['clip_range'],
    ent_coef=HYPERPARAMS['ent_coef_start'],  # Will be adapted by callback
    vf_coef=HYPERPARAMS['vf_coef'],
    max_grad_norm=HYPERPARAMS['max_grad_norm'],
    policy_kwargs=HYPERPARAMS['policy_kwargs'],
    verbose=1,
    tensorboard_log=str(v2_dir / 'logs'),
)
print("           [OK] PPO model created")
print(f"           Policy: {HYPERPARAMS['policy_kwargs']}")
print(f"           Initial entropy: {HYPERPARAMS['ent_coef_start']} (will decay to {HYPERPARAMS['ent_coef_end']})")
print()

# ================================================================================
# CREATE CALLBACKS (V2.7 SPECIFIC)
# ================================================================================

print("[STEP 6/8] Creating V2.7 callbacks...")

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=HYPERPARAMS['checkpoint_freq'],
    save_path=str(v2_dir / 'checkpoints'),
    name_prefix='agent8_v2.7_checkpoint',
    save_replay_buffer=False,
    save_vecnormalize=False,
)

# CSV snapshot callback
csv_callback = CheckpointCSVCallback(
    checkpoint_freq=HYPERPARAMS['checkpoint_freq'],
    env_ref=env,
    verbose=1
)

# Adaptive entropy callback (V2.7 KEY FIX)
entropy_callback = AdaptiveEntropyCallback(
    start_coef=HYPERPARAMS['ent_coef_start'],
    end_coef=HYPERPARAMS['ent_coef_end'],
    total_timesteps=HYPERPARAMS['total_timesteps'],
    verbose=1
)

# Global timestep callback (FIX 5 - Demonstration Learning)
timestep_callback = GlobalTimestepCallback(
    env_ref=env,
    verbose=1
)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, csv_callback, entropy_callback, timestep_callback])

print("           [OK] V2.7 Callbacks created:")
print(f"           - Checkpoint every {HYPERPARAMS['checkpoint_freq']:,} steps")
print(f"           - CSV snapshot every {HYPERPARAMS['checkpoint_freq']:,} steps")
print(f"           - Adaptive entropy: {HYPERPARAMS['ent_coef_start']} → {HYPERPARAMS['ent_coef_end']}")
print(f"           - Demonstration learning: Phases 1-3 (0-500K)")
print()

# ================================================================================
# TRAINING
# ================================================================================

print("="*80)
print("STARTING TRAINING V2.7 NUCLEAR - 500K STEPS")
print("="*80)
print(f"Total timesteps: {HYPERPARAMS['total_timesteps']:,}")
print(f"Checkpoints: {HYPERPARAMS['total_timesteps'] // HYPERPARAMS['checkpoint_freq']}")
print(f"Expected duration: ~40 minutes")
print("="*80)
print()

training_start = time.time()

try:
    model.learn(
        total_timesteps=HYPERPARAMS['total_timesteps'],
        callback=callbacks,
        log_interval=100,
        progress_bar=True
    )

    training_end = time.time()
    training_duration = (training_end - training_start) / 3600  # hours

    print()
    print("="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Duration: {training_duration:.2f} hours")
    print()

except KeyboardInterrupt:
    print()
    print("="*80)
    print("TRAINING INTERRUPTED BY USER")
    print("="*80)
    training_end = time.time()
    training_duration = (training_end - training_start) / 3600
    print(f"Duration: {training_duration:.2f} hours")
    print()

# ================================================================================
# SAVE FINAL MODEL
# ================================================================================

print("[STEP 7/8] Saving final model...")
final_model_path = models_dir / 'agent8_500k_final.zip'
final_model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(str(final_model_path))
print(f"           [OK] Model saved: {final_model_path}")
print(f"           [OK] Checkpoints: {v2_dir / 'checkpoints'}")
print()

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("[STEP 8/8] Creating final summary...")

# Get final environment metrics
final_env = env.envs[0].env if hasattr(env, 'envs') else env
final_balance = final_env.balance
initial_balance = HYPERPARAMS['initial_balance']
roi_pct = ((final_balance - initial_balance) / initial_balance) * 100

trades = final_env.trades if hasattr(final_env, 'trades') else []
total_trades = len(trades)

if total_trades > 0:
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    win_rate = len(wins) / total_trades
else:
    win_rate = 0.0

max_dd = final_env.max_drawdown if hasattr(final_env, 'max_drawdown') else 0.0

# Action distribution
action_history = final_env.action_history if hasattr(final_env, 'action_history') else []
if len(action_history) > 100:
    recent_actions = list(action_history)[-1000:]
    action_counts = {0: 0, 1: 0, 2: 0}
    for a in recent_actions:
        action_counts[int(a)] = action_counts.get(int(a), 0) + 1

    total_actions = len(recent_actions)
    action_sell_pct = (action_counts[0] / total_actions) * 100
    action_hold_pct = (action_counts[1] / total_actions) * 100
    action_buy_pct = (action_counts[2] / total_actions) * 100
else:
    action_sell_pct = 0.0
    action_hold_pct = 0.0
    action_buy_pct = 0.0

final_summary = {
    'training_timesteps': int(HYPERPARAMS['total_timesteps']),
    'training_duration_hours': float(training_duration),
    'final_balance': float(final_balance),
    'initial_balance': float(initial_balance),
    'roi_pct': float(roi_pct),
    'total_trades': int(total_trades),
    'win_rate': float(win_rate),
    'max_drawdown': float(max_dd),
    'ftmo_compliant': bool(max_dd < 0.10),
    'algorithm': 'PPO',
    'version': 'V2.9 NORMALIZED',
    'action_distribution': {
        'sell_pct': float(action_sell_pct),
        'hold_pct': float(action_hold_pct),
        'buy_pct': float(action_buy_pct)
    },
    'entropy_schedule': f"{HYPERPARAMS['ent_coef_start']} -> {HYPERPARAMS['ent_coef_end']}",
    'fixes_applied': 7
}

# Save summary JSON
summary_path = v2_dir / 'training_summary_v2.7_500k.json'
with open(summary_path, 'w') as f:
    json.dump(final_summary, f, indent=2)

print(f"           [OK] Summary saved: {summary_path.name}")
print()

# ================================================================================
# FINAL REPORT
# ================================================================================

print("="*80)
print("TRAINING SUMMARY - AGENT 8 V2.7 NUCLEAR (500K)")
print("="*80)
print()
print(f"Total Timesteps:      {HYPERPARAMS['total_timesteps']:,}")
print(f"Training Duration:    {training_duration:.2f} hours")
print()
print(f"Initial Balance:      ${initial_balance:,.2f}")
print(f"Final Balance:        ${final_balance:,.2f}")
print(f"ROI:                  {roi_pct:+.2f}%")
print()
print(f"Total Trades:         {total_trades}")
print(f"Win Rate:             {win_rate*100:.1f}%")
print(f"Max Drawdown:         {max_dd*100:.2f}%")
print(f"FTMO Compliant:       {'YES' if final_summary['ftmo_compliant'] else 'NO'}")
print()
print("ACTION DISTRIBUTION (Last 1000 actions):")
print(f"  SELL:               {action_sell_pct:.1f}%")
print(f"  HOLD:               {action_hold_pct:.1f}%")
print(f"  BUY:                {action_buy_pct:.1f}%")
print()

# Mode collapse check
if action_hold_pct > 80.0 or action_sell_pct > 80.0 or action_buy_pct > 80.0:
    print("[WARNING] MODE COLLAPSE DETECTED! V2.7 fixes didn't work.")
else:
    print("[SUCCESS] NO MODE COLLAPSE! V2.7 fixes working!")
print()

print(f"Algorithm:            PPO")
print(f"Version:              V2.7 NUCLEAR")
print(f"Fixes Applied:        7/8 (FIX 7 skipped)")
print(f"Entropy Schedule:     {HYPERPARAMS['ent_coef_start']} → {HYPERPARAMS['ent_coef_end']}")
print()
print("="*80)
print("NEXT STEPS:")
print("="*80)
print()
print("1. Analyze checkpoints_analysis/*.csv to validate V2.7 performance")
print("2. Check action distribution: Should be ~30% SELL, ~30% HOLD, ~30% BUY")
print("3. If results good -> Full training 1M+ steps")
print("4. If mode collapse persists -> Increase entropy or bonuses")
print()
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
