r"""
FULL DIAGNOSTIC - Agent 8 Complete Analysis
================================================================================
Tests complets pour comprendre pourquoi l'agent ne trade pas

TESTS INCLUS:
  1. Distribution des actions (HOLD/BUY/SELL %)
  2. Random vs Trained comparison
  3. Observations normalization check
  4. Reward per action type
  5. Exploration (entropy) analysis
  6. Reward cumulatif analysis
  7. Solutions recommendations

Usage:
  python analysis/full_diagnostic.py
  python analysis/full_diagnostic.py --model path/to/model.zip
  python analysis/full_diagnostic.py --episodes 50

Duration: ~5-10 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime
from collections import defaultdict
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
print("FULL DIAGNOSTIC - Agent 8 Complete Analysis")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# ================================================================================
# ARGUMENTS
# ================================================================================

parser = argparse.ArgumentParser(description='Full diagnostic for Agent 8')
parser.add_argument('--model', type=str, default=None, help='Path to model.zip')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
parser.add_argument('--steps_per_episode', type=int, default=500, help='Steps per episode')
args = parser.parse_args()

# ================================================================================
# IMPORTS
# ================================================================================

print("[SETUP] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("        [OK] Environment modules")
except ImportError as e:
    print(f"        [ERROR] {e}")
    sys.exit(1)

try:
    import torch
    from stable_baselines3 import PPO
    print("        [OK] Stable-Baselines3")
    HAS_SB3 = True
except ImportError:
    print("        [WARNING] SB3 not available")
    HAS_SB3 = False

print()

# ================================================================================
# LOAD DATA
# ================================================================================

print("[SETUP] Loading data...")
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

print(f"        [OK] Data: {features_df.shape}")
print()

# ================================================================================
# CREATE ENVIRONMENT
# ================================================================================

print("[SETUP] Creating environment...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_episode_steps=args.steps_per_episode,
    verbose=False,
    training_mode=True
)
print(f"        [OK] Environment ready")
print()

# ================================================================================
# LOAD MODEL (if available)
# ================================================================================

model = None
if HAS_SB3:
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
                print(f"[SETUP] Model loaded: {path}")
                break
            except Exception as e:
                pass

    if model is None:
        print("[SETUP] No model found - comparing with random only")

print()

action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

# ================================================================================
# TEST 1: DISTRIBUTION DES ACTIONS
# ================================================================================

print("=" * 80)
print("TEST 1: DISTRIBUTION DES ACTIONS")
print("=" * 80)
print()

def run_episodes(env, policy='random', model=None, n_episodes=10, label=""):
    """Run episodes and collect statistics."""
    action_counts = {0: 0, 1: 0, 2: 0}
    total_rewards = []
    episode_rewards = []
    positions_opened = 0
    positions_closed = 0
    rewards_per_action = {0: [], 1: [], 2: []}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(env, 'set_global_timestep'):
            env.set_global_timestep(50000)

        ep_reward = 0
        for step in range(args.steps_per_episode):
            if policy == 'random':
                action = np.random.choice([0, 1, 2])
            elif policy == 'model' and model is not None:
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                action = np.random.choice([0, 1, 2])

            obs, reward, done, truncated, info = env.step(action)

            action_counts[action] += 1
            rewards_per_action[action].append(reward)
            ep_reward += reward
            total_rewards.append(reward)

            if env.position_opened_this_step:
                positions_opened += 1
            if env.position_closed_this_step:
                positions_closed += 1

            if done or truncated:
                break

        episode_rewards.append(ep_reward)

    return {
        'action_counts': action_counts,
        'total_rewards': total_rewards,
        'episode_rewards': episode_rewards,
        'positions_opened': positions_opened,
        'positions_closed': positions_closed,
        'rewards_per_action': rewards_per_action
    }

# Run random baseline
print(f"Running {args.episodes} episodes with RANDOM actions...")
random_stats = run_episodes(env, policy='random', n_episodes=args.episodes)

# Run model if available
model_stats = None
if model is not None:
    print(f"Running {args.episodes} episodes with MODEL actions...")
    model_stats = run_episodes(env, policy='model', model=model, n_episodes=args.episodes)

print()
print("RANDOM Agent Distribution:")
print("-" * 50)
total = sum(random_stats['action_counts'].values())
for action, count in random_stats['action_counts'].items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {action_names[action]}: {count:5d} ({pct:5.1f}%) {bar}")

if model_stats:
    print()
    print("TRAINED Model Distribution:")
    print("-" * 50)
    total = sum(model_stats['action_counts'].values())
    for action, count in model_stats['action_counts'].items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {action_names[action]}: {count:5d} ({pct:5.1f}%) {bar}")

    # Check for HOLD dominance
    hold_pct = model_stats['action_counts'][1] / total * 100
    if hold_pct > 80:
        print()
        print("[CRITICAL] HOLD > 80% - Agent a PEUR de trader!")
    elif hold_pct > 60:
        print()
        print("[WARNING] HOLD > 60% - Agent passif")

print()

# ================================================================================
# TEST 2: RANDOM VS TRAINED COMPARISON
# ================================================================================

print("=" * 80)
print("TEST 2: RANDOM VS TRAINED COMPARISON")
print("=" * 80)
print()

print("RANDOM Agent:")
print(f"  Avg episode reward: {np.mean(random_stats['episode_rewards']):+.2f}")
print(f"  Avg step reward:    {np.mean(random_stats['total_rewards']):+.4f}")
print(f"  Positions opened:   {random_stats['positions_opened']}")
print(f"  Positions closed:   {random_stats['positions_closed']}")

if model_stats:
    print()
    print("TRAINED Model:")
    print(f"  Avg episode reward: {np.mean(model_stats['episode_rewards']):+.2f}")
    print(f"  Avg step reward:    {np.mean(model_stats['total_rewards']):+.4f}")
    print(f"  Positions opened:   {model_stats['positions_opened']}")
    print(f"  Positions closed:   {model_stats['positions_closed']}")

    # Compare
    random_reward = np.mean(random_stats['episode_rewards'])
    trained_reward = np.mean(model_stats['episode_rewards'])

    print()
    if abs(trained_reward - random_reward) < abs(random_reward) * 0.1:
        print("[CRITICAL] Random ~ Trained")
        print("           L'agent n'a rien appris d'utile!")
    elif trained_reward > random_reward:
        improvement = (trained_reward - random_reward) / abs(random_reward) * 100 if random_reward != 0 else 0
        print(f"[OK] Trained > Random (+{improvement:.1f}%)")
        print("     L'agent a appris quelque chose")
    else:
        print("[WARNING] Random > Trained")
        print("          L'agent performe MOINS bien qu'aleatoire!")

print()

# ================================================================================
# TEST 3: OBSERVATIONS NORMALIZATION
# ================================================================================

print("=" * 80)
print("TEST 3: OBSERVATIONS NORMALIZATION CHECK")
print("=" * 80)
print()

obs, _ = env.reset()

print("Observation shape:", obs.shape)
print()
print("Statistics:")
print(f"  Min:    {obs.min():.4f}")
print(f"  Max:    {obs.max():.4f}")
print(f"  Mean:   {obs.mean():.4f}")
print(f"  Std:    {obs.std():.4f}")
print()

# Check for problems
zeros = (obs == 0).sum()
near_zero = (np.abs(obs) < 0.001).sum()
large = (np.abs(obs) > 10).sum()
very_large = (np.abs(obs) > 100).sum()

print("Problematic values:")
print(f"  Exactly 0:      {zeros}/{len(obs)} ({zeros/len(obs)*100:.1f}%)")
print(f"  Near 0 (<0.001): {near_zero}/{len(obs)} ({near_zero/len(obs)*100:.1f}%)")
print(f"  Large (>10):    {large}/{len(obs)} ({large/len(obs)*100:.1f}%)")
print(f"  Very large (>100): {very_large}/{len(obs)} ({very_large/len(obs)*100:.1f}%)")

print()

if very_large > 0:
    print("[CRITICAL] Valeurs > 100 detectees!")
    print("           -> Features mal normalisees (prix bruts?)")
    print("           -> SOLUTION: Normaliser toutes les features [-1, 1] ou [0, 1]")
elif large / len(obs) > 0.1:
    print("[WARNING] Plus de 10% de valeurs > 10")
    print("          -> Verifier la normalisation")
elif zeros / len(obs) > 0.5:
    print("[WARNING] Plus de 50% de valeurs = 0")
    print("          -> Features peu informatives")
else:
    print("[OK] Observations semblent correctement normalisees")

print()

# ================================================================================
# TEST 4: REWARD PER ACTION TYPE
# ================================================================================

print("=" * 80)
print("TEST 4: REWARD PAR TYPE D'ACTION")
print("=" * 80)
print()

stats = random_stats  # Use random as baseline

print("Reward cumule par action (RANDOM baseline):")
print("-" * 60)
for action, rewards in stats['rewards_per_action'].items():
    if rewards:
        avg = np.mean(rewards)
        total = np.sum(rewards)
        count = len(rewards)
        print(f"  {action_names[action]}: mean={avg:+.4f}, total={total:+.2f}, count={count}")

print()

# Check if HOLD is more rewarding
hold_rewards = stats['rewards_per_action'][1]
trade_rewards = stats['rewards_per_action'][0] + stats['rewards_per_action'][2]

if hold_rewards and trade_rewards:
    avg_hold = np.mean(hold_rewards)
    avg_trade = np.mean(trade_rewards)

    print(f"Comparaison:")
    print(f"  Avg HOLD reward:  {avg_hold:+.4f}")
    print(f"  Avg TRADE reward: {avg_trade:+.4f}")
    print()

    if avg_hold > avg_trade:
        print("[WARNING] HOLD plus recompense que TRADE")
        print("          -> L'agent va preferer ne rien faire")
    else:
        print("[OK] TRADE >= HOLD")

print()

# ================================================================================
# TEST 5: EXPLORATION / ENTROPY ANALYSIS
# ================================================================================

print("=" * 80)
print("TEST 5: EXPLORATION / ENTROPY")
print("=" * 80)
print()

if model is not None and HAS_SB3:
    print("Analyzing model entropy...")

    obs, _ = env.reset()
    entropies = []

    for _ in range(100):
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)

        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.numpy()[0]

            # Shannon entropy
            probs_clipped = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs_clipped * np.log(probs_clipped))
            entropies.append(entropy)

        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, truncated, _ = env.step(int(action))
        if done or truncated:
            obs, _ = env.reset()

    avg_entropy = np.mean(entropies)
    max_entropy = np.log(3)  # Max for 3 actions

    print(f"  Average entropy: {avg_entropy:.3f}")
    print(f"  Max entropy:     {max_entropy:.3f}")
    print(f"  Ratio:           {avg_entropy/max_entropy*100:.1f}%")
    print()

    if avg_entropy < 0.3:
        print("[CRITICAL] Entropy TRES BASSE (< 0.3)")
        print("           -> Agent trop deterministe")
        print("           -> SOLUTION: Augmenter ent_coef (0.01 -> 0.1)")
    elif avg_entropy < 0.6:
        print("[WARNING] Entropy BASSE (< 0.6)")
        print("          -> Exploration insuffisante")
    else:
        print("[OK] Entropy suffisante")
else:
    print("[INFO] Model non disponible - skip entropy analysis")
    print("       Pour PPO, verifier ent_coef dans les hyperparametres")

print()

# Check env entropy coefficient if accessible
if hasattr(env, 'entropy_coef'):
    print(f"Environment entropy_coef: {env.entropy_coef}")

print()

# ================================================================================
# TEST 6: POSITIONS ANALYSIS
# ================================================================================

print("=" * 80)
print("TEST 6: POSITIONS ANALYSIS")
print("=" * 80)
print()

print(f"Sur {args.episodes} episodes x {args.steps_per_episode} steps:")
print()
print("RANDOM:")
print(f"  Positions opened: {random_stats['positions_opened']}")
print(f"  Positions closed: {random_stats['positions_closed']}")
if random_stats['positions_opened'] > 0:
    close_rate = random_stats['positions_closed'] / random_stats['positions_opened'] * 100
    print(f"  Close rate:       {close_rate:.1f}%")

if model_stats:
    print()
    print("MODEL:")
    print(f"  Positions opened: {model_stats['positions_opened']}")
    print(f"  Positions closed: {model_stats['positions_closed']}")
    if model_stats['positions_opened'] > 0:
        close_rate = model_stats['positions_closed'] / model_stats['positions_opened'] * 100
        print(f"  Close rate:       {close_rate:.1f}%")

    if model_stats['positions_opened'] == 0:
        print()
        print("[CRITICAL] Model n'ouvre AUCUNE position!")
        print("           -> Verifier reward function")
        print("           -> Verifier FIX 8 (over-trading protection)")

print()

# ================================================================================
# SYNTHESE ET RECOMMANDATIONS
# ================================================================================

print("=" * 80)
print("SYNTHESE ET RECOMMANDATIONS")
print("=" * 80)
print()

problems = []
solutions = []

# Check action distribution
if model_stats:
    hold_pct = model_stats['action_counts'][1] / sum(model_stats['action_counts'].values()) * 100
    if hold_pct > 80:
        problems.append("Agent 95%+ HOLD (peur de trader)")
        solutions.append("Ajouter penalite HOLD: reward -= 0.01 si HOLD sans position")

# Check random vs trained
if model_stats:
    if abs(np.mean(model_stats['episode_rewards']) - np.mean(random_stats['episode_rewards'])) < 1:
        problems.append("Agent ~ Random (n'a rien appris)")
        solutions.append("Revoir architecture ou hyperparametres")

# Check observations
if very_large > 0:
    problems.append("Features mal normalisees (valeurs > 100)")
    solutions.append("Normaliser toutes les features en [-1, 1]")

# Check positions
if model_stats and model_stats['positions_opened'] == 0:
    problems.append("0 positions ouvertes")
    solutions.append("Ajouter reward +5.0 a l'ouverture de position")

# Check HOLD reward
if hold_rewards and trade_rewards:
    if np.mean(hold_rewards) > np.mean(trade_rewards):
        problems.append("HOLD reward > TRADE reward")
        solutions.append("Inverser: penaliser HOLD, recompenser TRADE")

print("PROBLEMES DETECTES:")
print("-" * 50)
if problems:
    for i, prob in enumerate(problems, 1):
        print(f"  {i}. {prob}")
else:
    print("  Aucun probleme majeur detecte")

print()
print("SOLUTIONS RECOMMANDEES:")
print("-" * 50)
if solutions:
    for i, sol in enumerate(solutions, 1):
        print(f"  {i}. {sol}")
else:
    print("  Aucune action requise")

print()

# Final recommendation
print("=" * 80)
if len(problems) >= 3:
    print("VERDICT: AGENT NECESSITE REFONTE")
    print("         Plusieurs problemes critiques detectes")
elif len(problems) >= 1:
    print("VERDICT: AGENT A CORRIGER")
    print("         Appliquer les solutions avant training long")
else:
    print("VERDICT: AGENT SEMBLE OK")
    print("         Pret pour training long")
print("=" * 80)
print()
print("FULL DIAGNOSTIC TERMINE")
