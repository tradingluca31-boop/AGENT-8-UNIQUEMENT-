r"""
CHECK Q-VALUES / POLICY OUTPUT - Agent 8
================================================================================
Regarde ce que l'agent "pense" vraiment

Pour PPO: Affiche les probabilites d'action (policy output)
Pour DQN: Affiche les Q-values pour chaque action

Si P(HOLD) >> P(BUY) et P(SELL) -> L'agent a appris que ne rien faire est "safe"

Usage:
  python analysis/check_qvalues.py
  python analysis/check_qvalues.py --model path/to/model.zip
  python analysis/check_qvalues.py --n_samples 100

Duration: ~2 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
analysis_dir = Path(__file__).resolve().parent
agent8_root = analysis_dir.parent
env_dir = agent8_root / "environment"
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

sys.path.insert(0, str(goldrl_root))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(env_dir))

print("=" * 80)
print("CHECK Q-VALUES / POLICY OUTPUT - Agent 8")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# ================================================================================
# ARGUMENTS
# ================================================================================

parser = argparse.ArgumentParser(description='Check Q-values / Policy output')
parser.add_argument('--model', type=str, default=None, help='Path to model.zip')
parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to analyze')
args = parser.parse_args()

# ================================================================================
# IMPORTS
# ================================================================================

print("[1/5] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("       [OK] Environment modules")
except ImportError as e:
    print(f"       [ERROR] {e}")
    sys.exit(1)

try:
    import torch
    from stable_baselines3 import PPO
    print("       [OK] Stable-Baselines3 + PyTorch")
    HAS_SB3 = True
except ImportError:
    print("       [WARNING] SB3 not available - limited analysis")
    HAS_SB3 = False

print()

# ================================================================================
# LOAD DATA
# ================================================================================

print("[2/5] Loading data...")
df_full, auxiliary_data = load_data_agent8()
df_full = df_full.loc['2008-01-01':'2020-12-31']

for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc['2008-01-01':'2020-12-31']

features_df = calculate_all_features(df_full, auxiliary_data)
prices_raw = auxiliary_data['xauusd_raw']['H1'].loc['2008-01-01':'2020-12-31'].copy()
common_index = features_df.index.intersection(prices_raw.index)
prices_df = prices_raw.loc[common_index]
features_df = features_df.loc[common_index]

print(f"       [OK] Data loaded: {features_df.shape}")
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
    verbose=False,
    training_mode=True
)
print(f"       [OK] Environment created")
print()

# ================================================================================
# LOAD MODEL (if available)
# ================================================================================

model = None
if HAS_SB3:
    print("[4/5] Loading model...")

    # Try to find model
    model_paths = [
        args.model,
        agent8_root / "models" / "best_model.zip",
        agent8_root / "checkpoints" / "agent8_500k_final.zip",
        agent8_root / "checkpoints" / "agent8_checkpoint_50000_steps.zip",
    ]

    for path in model_paths:
        if path and Path(path).exists():
            try:
                model = PPO.load(path, env=env)
                print(f"       [OK] Model loaded: {path}")
                break
            except Exception as e:
                print(f"       [WARNING] Failed to load {path}: {e}")

    if model is None:
        print("       [INFO] No model found - will analyze random policy")
else:
    print("[4/5] Skipping model loading (SB3 not available)")

print()

# ================================================================================
# ANALYZE POLICY OUTPUT
# ================================================================================

print("[5/5] Analyzing policy output...")
print()

action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
n_samples = args.n_samples

# Storage for analysis
all_probs = []
all_actions = []
all_values = []

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

print("=" * 80)
print(f"POLICY OUTPUT ANALYSIS ({n_samples} samples)")
print("=" * 80)
print()

for i in range(n_samples):
    if model is not None:
        # Get policy distribution
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)

        with torch.no_grad():
            # Get action distribution from policy
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.numpy()[0]

            # Get value estimate
            value = model.policy.predict_values(obs_tensor).numpy()[0][0]

            # Get action
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)

        all_probs.append(probs)
        all_values.append(value)
        all_actions.append(action)

        # Show first 10 samples in detail
        if i < 10:
            print(f"Sample {i+1:2d}:")
            print(f"  Probabilities: SELL={probs[0]:.3f}, HOLD={probs[1]:.3f}, BUY={probs[2]:.3f}")
            print(f"  Value estimate: {value:.2f}")
            print(f"  Action chosen: {action_names[action]}")
            print()
    else:
        # Random policy baseline
        probs = np.array([0.33, 0.34, 0.33])
        action = np.random.choice([0, 1, 2])
        all_probs.append(probs)
        all_actions.append(action)

    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

# ================================================================================
# AGGREGATE STATISTICS
# ================================================================================

print("=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)
print()

all_probs = np.array(all_probs)
all_actions = np.array(all_actions)

# Average probabilities
avg_probs = all_probs.mean(axis=0)
std_probs = all_probs.std(axis=0)

print("Average Action Probabilities:")
print("-" * 50)
for i, (avg, std) in enumerate(zip(avg_probs, std_probs)):
    bar_len = int(avg * 40)
    bar = "█" * bar_len
    print(f"  {action_names[i]}: {avg:.3f} ± {std:.3f}  {bar}")

print()

# Action distribution
action_counts = np.bincount(all_actions, minlength=3)
action_pcts = action_counts / len(all_actions) * 100

print("Actual Actions Taken:")
print("-" * 50)
for i, (count, pct) in enumerate(zip(action_counts, action_pcts)):
    bar_len = int(pct / 2.5)
    bar = "█" * bar_len
    print(f"  {action_names[i]}: {count:3d} ({pct:5.1f}%)  {bar}")

print()

# Value estimates (if model available)
if all_values:
    all_values = np.array(all_values)
    print("Value Estimates:")
    print("-" * 50)
    print(f"  Mean:   {all_values.mean():.2f}")
    print(f"  Std:    {all_values.std():.2f}")
    print(f"  Min:    {all_values.min():.2f}")
    print(f"  Max:    {all_values.max():.2f}")
    print()

# ================================================================================
# DIAGNOSIS
# ================================================================================

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()

# Check if HOLD dominates
hold_prob = avg_probs[1]
trade_prob = avg_probs[0] + avg_probs[2]

if hold_prob > 0.6:
    print("[CRITICAL] P(HOLD) = {:.1f}% >> P(TRADE) = {:.1f}%".format(hold_prob*100, trade_prob*100))
    print()
    print("L'agent a APPRIS que ne rien faire est 'SAFE'!")
    print()
    print("Causes possibles:")
    print("  1. Reward pour HOLD trop eleve (ou pas de penalite)")
    print("  2. Penalites pour pertes trop fortes -> peur de trader")
    print("  3. Entropy trop basse -> exploitation prematuree")
    print("  4. Demonstration Learning pas assez fort")
    print()
    print("Solutions:")
    print("  1. Augmenter penalite HOLD consecutive (-0.1 -> -0.5)")
    print("  2. Augmenter reward trading (+5.0 -> +10.0)")
    print("  3. Augmenter entropy (0.2 -> 0.4)")
    print("  4. Forcer plus de trades en Demonstration Learning")

elif avg_probs[0] > 0.7 or avg_probs[2] > 0.7:
    dominant = "SELL" if avg_probs[0] > avg_probs[2] else "BUY"
    print(f"[WARNING] MODE COLLAPSE detecte: {dominant} = {max(avg_probs[0], avg_probs[2])*100:.1f}%")
    print()
    print("L'agent est BLOQUE sur une seule action!")
    print()
    print("Solutions:")
    print("  1. Augmenter entropy coefficient")
    print("  2. Activer action masking")
    print("  3. Reset le training avec nouveaux hyperparametres")

elif abs(avg_probs[0] - avg_probs[2]) > 0.3:
    bias = "SELL" if avg_probs[0] > avg_probs[2] else "BUY"
    print(f"[WARNING] Bias detecte vers {bias}")
    print(f"          SELL={avg_probs[0]*100:.1f}% vs BUY={avg_probs[2]*100:.1f}%")
    print()
    print("L'agent prefere un cote du marche.")
    print("Peut etre OK si le marche est effectivement baissier/haussier.")

else:
    print("[OK] Distribution equilibree!")
    print(f"     SELL={avg_probs[0]*100:.1f}%, HOLD={avg_probs[1]*100:.1f}%, BUY={avg_probs[2]*100:.1f}%")
    print()
    print("L'agent explore les 3 actions de maniere equilibree.")

print()

# ================================================================================
# ENTROPY ANALYSIS
# ================================================================================

print("=" * 80)
print("ENTROPY ANALYSIS")
print("=" * 80)
print()

# Calculate Shannon entropy for each sample
def shannon_entropy(probs):
    """Calculate Shannon entropy."""
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    return -np.sum(probs * np.log(probs))

entropies = [shannon_entropy(p) for p in all_probs]
avg_entropy = np.mean(entropies)
max_entropy = shannon_entropy(np.array([1/3, 1/3, 1/3]))  # Maximum for 3 actions

print(f"Average Policy Entropy: {avg_entropy:.3f}")
print(f"Maximum Entropy (uniform): {max_entropy:.3f}")
print(f"Entropy Ratio: {avg_entropy/max_entropy*100:.1f}%")
print()

if avg_entropy < 0.5:
    print("[CRITICAL] Entropy TRES BASSE (< 0.5)")
    print("           L'agent est trop deterministe, n'explore plus.")
elif avg_entropy < 0.8:
    print("[WARNING] Entropy BASSE (< 0.8)")
    print("          L'agent commence a se figer sur certaines actions.")
else:
    print("[OK] Entropy SAINE (>= 0.8)")
    print("     L'agent explore suffisamment.")

print()
print("=" * 80)
print("CHECK Q-VALUES / POLICY TERMINE")
print("=" * 80)
