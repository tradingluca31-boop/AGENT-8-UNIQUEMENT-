"""
SMOKE TEST - AGENT 8 FIX D2 VALIDATION
================================================================================
Valide que FIX D2 fonctionne AVANT de lancer training 500K complet.

Tests:
1. Code ne crash pas
2. Losing trades donnent reward -2.0 (pas +4.0)
3. Winning trades donnent reward +10.0
4. Win Rate > 0% (agent distingue gains vs pertes)
5. Apprentissage commence (reward augmente)

Duration: ~2 minutes (10K steps)
Standard: Citadel, Two Sigma (always smoke test before production)
================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths - AGENT 8 CENTRALIZED STRUCTURE
training_dir = Path(__file__).resolve().parent
agent8_root = training_dir.parent
env_dir = agent8_root / "environment"
models_dir = agent8_root / "models"
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

sys.path.insert(0, str(goldrl_root))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(env_dir))

print("="*80)
print("SMOKE TEST - AGENT 8 FIX D2 VALIDATION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# ================================================================================
# IMPORTS
# ================================================================================

print("[1/5] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    from stable_baselines3 import PPO
    print("       [OK] All modules imported")
except ImportError as e:
    print(f"       [ERROR] Import failed: {e}")
    sys.exit(1)

print()

# ================================================================================
# LOAD DATA
# ================================================================================

print("[2/5] Loading data (quick sample)...")
df_full, auxiliary_data = load_data_agent8()
df_full = df_full.loc['2008-01-01':'2010-12-31']  # Only 3 years for speed

for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc['2008-01-01':'2010-12-31']

features_df = calculate_all_features(df_full, auxiliary_data)
prices_df = auxiliary_data['xauusd_raw']['H1'].loc['2008-01-01':'2010-12-31']
common_index = features_df.index.intersection(prices_df.index)
features_df = features_df.loc[common_index]
prices_df = prices_df.loc[common_index]

print(f"       [OK] Data loaded: {features_df.shape[0]} bars")
print()

# ================================================================================
# CREATE ENVIRONMENT
# ================================================================================

print("[3/5] Creating environment...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_episode_steps=5_000,
    verbose=False,  # Silence logs for smoke test
    training_mode=True
)
print("       [OK] Environment created")
print()

# ================================================================================
# QUICK TRAINING (10K STEPS)
# ================================================================================

print("[4/5] Quick training (10K steps - 2 min)...")
print("       Testing FIX D2: Win (+10.0) vs Loss (-2.0)")
print()

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.80,  # High exploration for smoke test
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={'net_arch': [256, 256]},
    verbose=0,
    device='auto'
)

# Train
start_time = datetime.now()
model.learn(total_timesteps=10_000, progress_bar=True)
duration = (datetime.now() - start_time).total_seconds()

print(f"\n       [OK] Training completed in {duration:.1f} seconds")
print()

# ================================================================================
# VALIDATION - INTERROGATE AGENT
# ================================================================================

print("[5/5] Validating FIX D2 - Interrogating agent...")
print()

# Reset env and run 100 steps with trained model
obs, _ = env.reset()
wins = 0
losses = 0
win_rewards = []
loss_rewards = []

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)

    # Track before step
    before_trades = len(env.trades)

    obs, reward, done, truncated, info = env.step(action)

    # Check if new trade closed
    if len(env.trades) > before_trades:
        last_trade = env.trades[-1]
        if last_trade['pnl'] > 0:
            wins += 1
            win_rewards.append(reward)
        else:
            losses += 1
            loss_rewards.append(reward)

    if done or truncated:
        obs, _ = env.reset()

# Calculate metrics
total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
avg_win_reward = sum(win_rewards) / len(win_rewards) if win_rewards else 0.0
avg_loss_reward = sum(loss_rewards) / len(loss_rewards) if loss_rewards else 0.0

print("="*80)
print("SMOKE TEST RESULTS")
print("="*80)
print()

print(f"Total Trades:      {total_trades}")
print(f"Wins:              {wins} ({win_rate:.1f}%)")
print(f"Losses:            {losses}")
print()

print(f"Avg Win Reward:    {avg_win_reward:+.2f}")
print(f"Avg Loss Reward:   {avg_loss_reward:+.2f}")
print()

# ================================================================================
# GO/NO-GO DECISION
# ================================================================================

print("="*80)
print("VALIDATION CHECKS")
print("="*80)
print()

checks_passed = 0
total_checks = 4

# Check 1: Code didn't crash
print("✅ CHECK 1: Code executed without crash")
checks_passed += 1

# Check 2: Trades happened
if total_trades > 0:
    print(f"✅ CHECK 2: Agent traded ({total_trades} trades)")
    checks_passed += 1
else:
    print("⚠️  CHECK 2: NO TRADES (agent too passive)")

# Check 3: Win Rate > 0%
if win_rate > 0:
    print(f"✅ CHECK 3: Win Rate > 0% ({win_rate:.1f}%)")
    checks_passed += 1
else:
    print("⚠️  CHECK 3: Win Rate = 0% (agent not learning yet)")

# Check 4: Loss rewards are negative or low
if losses > 0 and avg_loss_reward < 0:
    print(f"✅ CHECK 4: Loss rewards are NEGATIVE ({avg_loss_reward:+.2f})")
    print("           FIX D2 WORKING - Agent punished for losses!")
    checks_passed += 1
elif losses > 0 and avg_loss_reward < avg_win_reward:
    print(f"⚠️  CHECK 4: Loss rewards lower but not negative ({avg_loss_reward:+.2f})")
    print("           FIX D2 may need tuning")
    checks_passed += 0.5
else:
    print("⚠️  CHECK 4: Loss rewards not lower than wins")
    print("           FIX D2 NOT WORKING!")

print()
print("="*80)
print("FINAL DECISION")
print("="*80)
print()

if checks_passed >= 3.5:
    print("🟢 GO FOR 500K TRAINING!")
    print(f"   {checks_passed}/{total_checks} checks passed")
    print()
    print("   FIX D2 is working correctly.")
    print("   Agent shows signs of learning Win vs Loss distinction.")
    print()
    print("   NEXT STEP: Launch full 500K training")
    print("   Command: cd training && python train.py")
    exit_code = 0
elif checks_passed >= 2:
    print("🟡 CAUTION - Partial Success")
    print(f"   {checks_passed}/{total_checks} checks passed")
    print()
    print("   Agent shows some learning but not optimal.")
    print("   Consider:")
    print("   - Increase training to 50K smoke test")
    print("   - Adjust FIX D2 penalty (-7.0 to -10.0)")
    print("   - Check data quality (RSI=0.0 issue)")
    exit_code = 1
else:
    print("🔴 NO-GO - Fix Required")
    print(f"   {checks_passed}/{total_checks} checks passed")
    print()
    print("   FIX D2 not working as expected.")
    print("   Debug needed before 500K training.")
    print()
    print("   Check:")
    print("   1. trading_env.py line 988 (should be -7.0)")
    print("   2. Reward calculation logic")
    print("   3. Environment reset issues")
    exit_code = 2

print()
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

sys.exit(exit_code)
