r"""
CHECK REWARD FUNCTION - Agent 8 Diagnostic
================================================================================
Checklist de diagnostic pour comprendre pourquoi l'agent ne trade pas

QUESTIONS CRITIQUES:
  A) Est-ce que "HOLD" donne un reward (meme petit)?
  B) Est-ce que les trades perdants sont TROP penalises?
  C) Y a-t-il un reward UNIQUEMENT a la cloture du trade?

PIEGE CLASSIQUE:
  Si reward n'arrive qu'a la cloture ET pertes penalisees
  -> L'agent apprend: ne jamais ouvrir = jamais de perte = SAFE

Usage:
  python analysis/check_reward_function.py

Duration: ~3 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
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
print("CHECK REWARD FUNCTION - Agent 8 Diagnostic")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# ================================================================================
# IMPORTS
# ================================================================================

print("[1/6] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("       [OK] Environment modules")
except ImportError as e:
    print(f"       [ERROR] {e}")
    sys.exit(1)

print()

# ================================================================================
# LOAD DATA
# ================================================================================

print("[2/6] Loading data...")
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

print("[3/6] Creating environment...")
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
# QUESTION A: HOLD DONNE-T-IL UN REWARD?
# ================================================================================

print("=" * 80)
print("QUESTION A: Est-ce que HOLD donne un reward (meme petit)?")
print("=" * 80)
print()

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

hold_rewards = []
for _ in range(100):
    obs, reward, done, truncated, info = env.step(1)  # HOLD
    hold_rewards.append(reward)
    if done or truncated:
        obs, _ = env.reset()

avg_hold_reward = np.mean(hold_rewards)
min_hold_reward = np.min(hold_rewards)
max_hold_reward = np.max(hold_rewards)

print(f"100 actions HOLD consecutives:")
print(f"  Reward moyen:   {avg_hold_reward:+.4f}")
print(f"  Reward min:     {min_hold_reward:+.4f}")
print(f"  Reward max:     {max_hold_reward:+.4f}")
print()

if avg_hold_reward > 0:
    print("[CRITICAL] HOLD donne un reward POSITIF!")
    print("           -> L'agent est RECOMPENSE pour ne rien faire")
    print("           -> SOLUTION: Penaliser HOLD (reward = -0.01 ou moins)")
    hold_verdict = "PROBLEME"
elif avg_hold_reward == 0:
    print("[WARNING] HOLD donne reward = 0")
    print("          -> Neutre, mais l'agent peut preferer 'safe'")
    print("          -> SOLUTION: Ajouter petite penalite (-0.01)")
    hold_verdict = "NEUTRE"
else:
    print("[OK] HOLD est penalise (reward < 0)")
    print("     -> L'agent est POUSSE a agir")
    hold_verdict = "OK"

print()

# ================================================================================
# QUESTION B: TRADES PERDANTS TROP PENALISES?
# ================================================================================

print("=" * 80)
print("QUESTION B: Les trades perdants sont-ils TROP penalises?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

# Collect rewards for winning and losing trades
winning_rewards = []
losing_rewards = []
open_rewards = []

for i in range(500):
    # Force trades
    if env.position == 0:
        action = np.random.choice([0, 2])  # SELL or BUY
    else:
        action = 1  # HOLD to let position run

    obs, reward, done, truncated, info = env.step(action)

    if env.position_opened_this_step:
        open_rewards.append(reward)

    if env.position_closed_this_step:
        if env.last_closed_pnl > 0:
            winning_rewards.append(reward)
        else:
            losing_rewards.append(reward)

    if done or truncated:
        obs, _ = env.reset()

print("Analyse des rewards de trading:")
print("-" * 50)

if open_rewards:
    print(f"Ouverture de position:")
    print(f"  Reward moyen:   {np.mean(open_rewards):+.2f}")
    print(f"  Reward min/max: {np.min(open_rewards):+.2f} / {np.max(open_rewards):+.2f}")
else:
    print("  [WARNING] Aucune position ouverte!")

print()

if winning_rewards:
    print(f"Trades GAGNANTS ({len(winning_rewards)}):")
    print(f"  Reward moyen:   {np.mean(winning_rewards):+.2f}")
    print(f"  Reward min/max: {np.min(winning_rewards):+.2f} / {np.max(winning_rewards):+.2f}")
else:
    print("  [INFO] Aucun trade gagnant enregistre")

print()

if losing_rewards:
    print(f"Trades PERDANTS ({len(losing_rewards)}):")
    print(f"  Reward moyen:   {np.mean(losing_rewards):+.2f}")
    print(f"  Reward min/max: {np.min(losing_rewards):+.2f} / {np.max(losing_rewards):+.2f}")
else:
    print("  [INFO] Aucun trade perdant enregistre")

print()

# Compare magnitudes
if winning_rewards and losing_rewards:
    avg_win = abs(np.mean(winning_rewards))
    avg_loss = abs(np.mean(losing_rewards))
    ratio = avg_loss / avg_win if avg_win > 0 else float('inf')

    print(f"Ratio |perte| / |gain|: {ratio:.2f}")

    if ratio > 3:
        print("[CRITICAL] Pertes TROP penalisees (ratio > 3)!")
        print("           -> L'agent a PEUR de perdre")
        print("           -> Il prefere ne pas trader du tout")
        print("           -> SOLUTION: Reduire penalite pertes ou augmenter reward gains")
        loss_verdict = "PROBLEME"
    elif ratio > 2:
        print("[WARNING] Pertes assez penalisees (ratio > 2)")
        print("          -> Peut creer de l'aversion au risque")
        loss_verdict = "ATTENTION"
    else:
        print("[OK] Ratio equilibre (ratio <= 2)")
        loss_verdict = "OK"
else:
    print("[INFO] Pas assez de donnees pour comparer")
    loss_verdict = "INCONNU"

print()

# ================================================================================
# QUESTION C: REWARD UNIQUEMENT A LA CLOTURE?
# ================================================================================

print("=" * 80)
print("QUESTION C: Y a-t-il un reward UNIQUEMENT a la cloture?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

# Track rewards during different phases
rewards_no_position = []  # When no position
rewards_during_position = []  # While holding a position
rewards_at_close = []  # At position close

position_active = False
for i in range(500):
    if env.position == 0:
        action = np.random.choice([0, 2])  # Open
    else:
        action = 1  # Hold position

    obs, reward, done, truncated, info = env.step(action)

    if env.position == 0 and not env.position_closed_this_step:
        rewards_no_position.append(reward)
    elif env.position != 0:
        rewards_during_position.append(reward)

    if env.position_closed_this_step:
        rewards_at_close.append(reward)

    if done or truncated:
        obs, _ = env.reset()

print("Distribution des rewards par phase:")
print("-" * 50)

if rewards_no_position:
    avg = np.mean(rewards_no_position)
    print(f"Sans position:      avg={avg:+.3f} (n={len(rewards_no_position)})")
else:
    print("Sans position:      Pas de donnees")

if rewards_during_position:
    avg = np.mean(rewards_during_position)
    print(f"Pendant position:   avg={avg:+.3f} (n={len(rewards_during_position)})")
else:
    print("Pendant position:   Pas de donnees")

if rewards_at_close:
    avg = np.mean(rewards_at_close)
    print(f"A la cloture:       avg={avg:+.3f} (n={len(rewards_at_close)})")
else:
    print("A la cloture:       Pas de donnees")

print()

# Check for the classic trap
if rewards_at_close and rewards_no_position:
    close_magnitude = abs(np.mean(rewards_at_close))
    other_magnitude = abs(np.mean(rewards_no_position + rewards_during_position))

    if close_magnitude > other_magnitude * 5:
        print("[CRITICAL] PIEGE CLASSIQUE DETECTE!")
        print("           -> Rewards concentres a la cloture")
        print("           -> Si pertes penalisees: agent evite d'ouvrir = jamais de perte = SAFE")
        print("           -> SOLUTION: Ajouter reward a l'OUVERTURE (+5.0)")
        close_verdict = "PIEGE"
    else:
        print("[OK] Rewards distribues (pas uniquement a la cloture)")
        close_verdict = "OK"
else:
    print("[INFO] Pas assez de donnees")
    close_verdict = "INCONNU"

print()

# ================================================================================
# QUESTION D: RATIO RISK/REWARD DANS LES REWARDS
# ================================================================================

print("=" * 80)
print("QUESTION D: Le reward encourage-t-il le bon risk/reward?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

# Check if small wins are rewarded similar to big wins
small_wins = []  # < 0.25% profit
medium_wins = []  # 0.25% - 0.5%
big_wins = []  # > 0.5%

for i in range(500):
    if env.position == 0:
        action = np.random.choice([0, 2])
    else:
        action = 1

    obs, reward, done, truncated, info = env.step(action)

    if env.position_closed_this_step and env.last_closed_pnl > 0:
        pnl_pct = env.last_closed_pnl / env.initial_balance * 100
        if pnl_pct < 0.25:
            small_wins.append((pnl_pct, reward))
        elif pnl_pct < 0.5:
            medium_wins.append((pnl_pct, reward))
        else:
            big_wins.append((pnl_pct, reward))

    if done or truncated:
        obs, _ = env.reset()

print("Rewards par taille de gain:")
print("-" * 50)

if small_wins:
    avg_reward = np.mean([r for _, r in small_wins])
    avg_pnl = np.mean([p for p, _ in small_wins])
    print(f"Petits gains (<0.25%):   avg_pnl={avg_pnl:.3f}%, avg_reward={avg_reward:+.2f} (n={len(small_wins)})")

if medium_wins:
    avg_reward = np.mean([r for _, r in medium_wins])
    avg_pnl = np.mean([p for p, _ in medium_wins])
    print(f"Moyens gains (0.25-0.5%): avg_pnl={avg_pnl:.3f}%, avg_reward={avg_reward:+.2f} (n={len(medium_wins)})")

if big_wins:
    avg_reward = np.mean([r for _, r in big_wins])
    avg_pnl = np.mean([p for p, _ in big_wins])
    print(f"Gros gains (>0.5%):      avg_pnl={avg_pnl:.3f}%, avg_reward={avg_reward:+.2f} (n={len(big_wins)})")

print()

if small_wins and big_wins:
    small_avg = np.mean([r for _, r in small_wins])
    big_avg = np.mean([r for _, r in big_wins])

    if big_avg > small_avg * 1.5:
        print("[OK] Gros gains mieux recompenses que petits gains")
        rr_verdict = "OK"
    else:
        print("[WARNING] Gros gains pas assez recompenses")
        print("          -> L'agent n'est pas incite a laisser courir les gains")
        rr_verdict = "ATTENTION"
else:
    print("[INFO] Pas assez de donnees")
    rr_verdict = "INCONNU"

print()

# ================================================================================
# SYNTHESE
# ================================================================================

print("=" * 80)
print("SYNTHESE - CHECKLIST REWARD FUNCTION")
print("=" * 80)
print()

checklist = [
    ("A. HOLD penalise?", hold_verdict),
    ("B. Pertes pas trop penalisees?", loss_verdict),
    ("C. Reward pas qu'a la cloture?", close_verdict),
    ("D. Risk/Reward encourage?", rr_verdict),
]

problems = 0
for question, verdict in checklist:
    if verdict == "OK":
        icon = "[OK]"
    elif verdict in ["PROBLEME", "PIEGE", "CRITICAL"]:
        icon = "[XX]"
        problems += 1
    elif verdict in ["ATTENTION", "WARNING", "NEUTRE"]:
        icon = "[!!]"
    else:
        icon = "[??]"

    print(f"  {icon} {question} -> {verdict}")

print()
print("-" * 50)

if problems >= 2:
    print("DIAGNOSTIC: REWARD FUNCTION PROBLEMATIQUE")
    print()
    print("L'agent a probablement appris que:")
    print("  'Ne pas trader = Jamais de perte = SAFE'")
    print()
    print("SOLUTIONS RECOMMANDEES:")
    print("  1. Ajouter reward +5.0 a l'OUVERTURE de position")
    print("  2. Ajouter penalite -0.01 pour HOLD (inaction)")
    print("  3. Reduire penalite pour pertes (ou augmenter reward gains)")
    print("  4. Ajouter bonus pour trades frequents (anti-passivite)")
elif problems == 1:
    print("DIAGNOSTIC: REWARD FUNCTION A SURVEILLER")
    print("Un probleme detecte - corriger avant training long")
else:
    print("DIAGNOSTIC: REWARD FUNCTION SEMBLE OK")
    print("Pas de probleme majeur detecte")

print()
print("=" * 80)
print("CHECK REWARD FUNCTION TERMINE")
print("=" * 80)
