r"""
BEHAVIORAL ANALYSIS - AGENT 8 PSYCHANALYSE
================================================================================
COMPRENDRE POURQUOI L'AGENT N'OUVRE PAS DE POSITIONS

QUESTIONS FONDAMENTALES:
  1. DISTRIBUTION: Que choisit l'agent ? (SELL/HOLD/BUY %)
  2. PEUR DU RISQUE: A-t-il peur d'ouvrir ? Préfère-t-il HOLD ?
  3. FERMETURE: Quand il ouvre, ferme-t-il ses positions ?
  4. REWARDS: Quels rewards reçoit-il pour chaque action ?
  5. LOGITS: Quelles probabilités le modèle attribue-t-il ?
  6. OBSERVATIONS: Que "voit" l'agent ? Les features sont-elles informatives ?
  7. EVOLUTION: Apprend-il à éviter le trading au fil du temps ?
  8. CONTEXTE: Dans quel contexte ouvre-t-il (si jamais) ?

Duration: ~5 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, deque
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
print("BEHAVIORAL ANALYSIS - AGENT 8 PSYCHANALYSE")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# ================================================================================
# IMPORTS
# ================================================================================

print("[1/8] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("       [OK] Environment modules")
except ImportError as e:
    print(f"       [ERROR] {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    print("       [OK] Stable-Baselines3")
except ImportError:
    print("       [WARNING] SB3 not available - will use random actions")
    PPO = None

print()

# ================================================================================
# LOAD DATA
# ================================================================================

print("[2/8] Loading data...")
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
print(f"       Price sample: ${prices_df['close'].iloc[1000]:.2f}")
print()

# ================================================================================
# CREATE ENVIRONMENT
# ================================================================================

print("[3/8] Creating environment...")
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
# QUESTION 1: DISTRIBUTION DES ACTIONS
# ================================================================================

print("=" * 80)
print("Q1: DISTRIBUTION DES ACTIONS - QUE CHOISIT L'AGENT ?")
print("=" * 80)
print()

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

# Simulate 1000 steps with random actions (or model if available)
action_counts = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY
action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

for _ in range(1000):
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, info = env.step(action)
    action_counts[action] += 1
    if done or truncated:
        obs, _ = env.reset()

total = sum(action_counts.values())
print("Distribution avec actions RANDOM (baseline):")
for action, count in action_counts.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"  {action_names[action]}: {pct:5.1f}% {bar}")

print()
print("NOTE: Avec random, devrait être ~33% chaque.")
print("      Si le modèle montre 90%+ HOLD, il a PEUR de trader.")
print()

# ================================================================================
# QUESTION 2: PEUR DU RISQUE - ANALYSE DU HOLD
# ================================================================================

print("=" * 80)
print("Q2: PEUR DU RISQUE - L'AGENT PRÉFÈRE-T-IL HOLD ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

# Track consecutive HOLDs
consecutive_holds = 0
max_consecutive_holds = 0
total_holds = 0
total_steps = 0
holds_after_trade = []  # HOLDs immediately after a trade action

last_action = None
for i in range(500):
    # Random action for now (or use model)
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, info = env.step(action)

    total_steps += 1

    if action == 1:  # HOLD
        consecutive_holds += 1
        total_holds += 1
        if last_action in [0, 2]:  # After SELL or BUY
            holds_after_trade.append(consecutive_holds)
    else:
        max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
        consecutive_holds = 0

    last_action = action

    if done or truncated:
        obs, _ = env.reset()

hold_pct = (total_holds / total_steps) * 100
avg_holds_after_trade = np.mean(holds_after_trade) if holds_after_trade else 0

print(f"Analyse sur 500 steps (random baseline):")
print(f"  Total HOLDs:              {total_holds}/{total_steps} ({hold_pct:.1f}%)")
print(f"  Max HOLDs consecutifs:    {max_consecutive_holds}")
print(f"  Moyenne HOLDs après trade: {avg_holds_after_trade:.1f}")
print()

if hold_pct > 50:
    print("[WARNING] HOLD > 50% - L'agent pourrait avoir peur de trader")
else:
    print("[OK] Distribution équilibrée")
print()

# ================================================================================
# QUESTION 3: FERMETURE DES POSITIONS
# ================================================================================

print("=" * 80)
print("Q3: FERMETURE - L'AGENT FERME-T-IL SES POSITIONS ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

positions_opened = 0
positions_closed = 0
position_durations = []
current_position_start = None

for i in range(1000):
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, info = env.step(action)

    if env.position_opened_this_step:
        positions_opened += 1
        current_position_start = i

    if env.position_closed_this_step:
        positions_closed += 1
        if current_position_start is not None:
            duration = i - current_position_start
            position_durations.append(duration)
            current_position_start = None

    if done or truncated:
        obs, _ = env.reset()
        current_position_start = None

avg_duration = np.mean(position_durations) if position_durations else 0
close_rate = (positions_closed / positions_opened * 100) if positions_opened > 0 else 0

print(f"Analyse sur 1000 steps:")
print(f"  Positions ouvertes:     {positions_opened}")
print(f"  Positions fermées:      {positions_closed}")
print(f"  Taux de fermeture:      {close_rate:.1f}%")
print(f"  Durée moyenne position: {avg_duration:.1f} steps")
print()

if positions_opened == 0:
    print("[CRITICAL] AUCUNE position ouverte!")
    print("           -> Problème d'ouverture (pas de fermeture à analyser)")
elif close_rate < 50:
    print("[WARNING] Taux de fermeture < 50%")
    print("          -> TP/SL trop loin ? ATR trop grand ?")
else:
    print("[OK] Positions se ferment correctement")
print()

# ================================================================================
# QUESTION 4: REWARDS PAR ACTION
# ================================================================================

print("=" * 80)
print("Q4: REWARDS - QUELS REWARDS REÇOIT L'AGENT ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

rewards_by_action = {0: [], 1: [], 2: []}  # SELL, HOLD, BUY

for i in range(500):
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, info = env.step(action)
    rewards_by_action[action].append(reward)

    if done or truncated:
        obs, _ = env.reset()

print("Rewards moyens par action:")
print("-" * 50)
for action, rewards in rewards_by_action.items():
    if rewards:
        avg_r = np.mean(rewards)
        min_r = np.min(rewards)
        max_r = np.max(rewards)
        std_r = np.std(rewards)
        print(f"  {action_names[action]}: avg={avg_r:+.2f}, min={min_r:+.2f}, max={max_r:+.2f}, std={std_r:.2f}")

print()
print("Interprétation:")
avg_hold = np.mean(rewards_by_action[1]) if rewards_by_action[1] else 0
avg_trade = np.mean(rewards_by_action[0] + rewards_by_action[2]) if (rewards_by_action[0] or rewards_by_action[2]) else 0

if avg_hold > avg_trade:
    print(f"  [WARNING] HOLD ({avg_hold:+.2f}) > TRADE ({avg_trade:+.2f})")
    print("            L'agent est RÉCOMPENSÉ pour ne rien faire!")
else:
    print(f"  [OK] TRADE ({avg_trade:+.2f}) >= HOLD ({avg_hold:+.2f})")
print()

# ================================================================================
# QUESTION 5: CONTEXTE D'OUVERTURE
# ================================================================================

print("=" * 80)
print("Q5: CONTEXTE - DANS QUEL CONTEXTE L'AGENT OUVRE-T-IL ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

open_contexts = []  # Store features when position opens

for i in range(1000):
    action = np.random.choice([0, 1, 2])

    # Get features BEFORE step
    current_features = env.features_df.iloc[env.current_step]
    current_price = env.prices_df['close'].iloc[env.current_step]

    obs, reward, done, truncated, info = env.step(action)

    if env.position_opened_this_step:
        # Record context
        context = {
            'step': i,
            'price': current_price,
            'action': action,
            'reward': reward,
        }

        # Get RSI if available
        rsi_cols = [c for c in env.features_df.columns if 'rsi' in c.lower()]
        if rsi_cols:
            context['rsi'] = current_features[rsi_cols[0]]

        open_contexts.append(context)

    if done or truncated:
        obs, _ = env.reset()

print(f"Positions ouvertes: {len(open_contexts)}")
print()

if open_contexts:
    print("Contexte des ouvertures:")
    print("-" * 60)
    for ctx in open_contexts[:10]:  # Show first 10
        action_name = action_names.get(ctx['action'], '?')
        rsi_str = f"RSI={ctx.get('rsi', 'N/A'):.1f}" if 'rsi' in ctx else ""
        print(f"  Step {ctx['step']:4d}: {action_name} @ ${ctx['price']:.2f} | reward={ctx['reward']:+.2f} | {rsi_str}")

    if len(open_contexts) > 10:
        print(f"  ... et {len(open_contexts) - 10} autres")
else:
    print("[CRITICAL] AUCUNE position ouverte sur 1000 steps!")
    print("           Le problème est dans l'OUVERTURE, pas la fermeture.")

print()

# ================================================================================
# QUESTION 6: ANALYSE DES OBSERVATIONS
# ================================================================================

print("=" * 80)
print("Q6: OBSERVATIONS - QUE 'VOIT' L'AGENT ?")
print("=" * 80)
print()

obs, _ = env.reset()

print(f"Shape observation: {obs.shape}")
print()

# Analyze observation values
print("Distribution des valeurs dans l'observation:")
print(f"  Min:    {obs.min():.4f}")
print(f"  Max:    {obs.max():.4f}")
print(f"  Mean:   {obs.mean():.4f}")
print(f"  Std:    {obs.std():.4f}")
print()

# Count special values
zeros = (obs == 0).sum()
near_zero = (np.abs(obs) < 0.01).sum()
large = (np.abs(obs) > 10).sum()

print(f"Valeurs spéciales:")
print(f"  Exactement 0:  {zeros}/{len(obs)} ({zeros/len(obs)*100:.1f}%)")
print(f"  Proche de 0:   {near_zero}/{len(obs)} ({near_zero/len(obs)*100:.1f}%)")
print(f"  Très grandes:  {large}/{len(obs)} ({large/len(obs)*100:.1f}%)")
print()

if zeros / len(obs) > 0.5:
    print("[WARNING] Plus de 50% des features sont à 0!")
    print("          L'agent ne reçoit pas assez d'information.")
elif large / len(obs) > 0.1:
    print("[WARNING] Plus de 10% des features sont très grandes (>10)!")
    print("          Problème de normalisation potentiel.")
else:
    print("[OK] Observations semblent normalisées correctement.")
print()

# ================================================================================
# QUESTION 7: ÉVOLUTION TEMPORELLE
# ================================================================================

print("=" * 80)
print("Q7: ÉVOLUTION - LE COMPORTEMENT CHANGE-T-IL ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

# Track behavior over time
window_size = 100
windows_data = []

for window in range(5):  # 5 windows of 100 steps
    window_actions = {0: 0, 1: 0, 2: 0}
    window_trades = 0

    for _ in range(window_size):
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = env.step(action)
        window_actions[action] += 1
        if env.position_opened_this_step:
            window_trades += 1
        if done or truncated:
            obs, _ = env.reset()

    hold_pct = window_actions[1] / window_size * 100
    windows_data.append({
        'window': window + 1,
        'hold_pct': hold_pct,
        'trades': window_trades
    })

print("Évolution par fenêtre de 100 steps (random baseline):")
print("-" * 50)
print("Fenêtre | HOLD %  | Trades")
print("-" * 50)
for w in windows_data:
    print(f"   {w['window']}    | {w['hold_pct']:5.1f}%  |   {w['trades']}")

print()
print("NOTE: Avec un modèle entraîné, on verrait si HOLD% augmente")
print("      au fil du temps (l'agent 'apprend' à ne pas trader).")
print()

# ================================================================================
# QUESTION 8: SYNTHÈSE COMPORTEMENTALE
# ================================================================================

print("=" * 80)
print("Q8: SYNTHÈSE - PROFIL COMPORTEMENTAL DE L'AGENT")
print("=" * 80)
print()

# Collect all findings
findings = []

# From Q1
if hold_pct > 60:
    findings.append("PASSIF: Préférence excessive pour HOLD")

# From Q3
if positions_opened == 0:
    findings.append("BLOQUÉ: N'ouvre jamais de positions")
elif close_rate < 30:
    findings.append("LENT: Ouvre mais ferme rarement")

# From Q4
if avg_hold > avg_trade:
    findings.append("MAL RÉCOMPENSÉ: HOLD plus profitable que TRADE")

# From Q6
if zeros / len(obs) > 0.5:
    findings.append("AVEUGLE: Trop de features à zéro")

print("DIAGNOSTIC COMPORTEMENTAL:")
print("-" * 60)

if findings:
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")
else:
    print("  Aucun problème comportemental majeur détecté.")

print()
print("=" * 80)
print("HYPOTHÈSES SUR LA 'PSYCHOLOGIE' DE L'AGENT:")
print("=" * 80)
print()

if positions_opened == 0:
    print("HYPOTHÈSE PRINCIPALE: L'agent est BLOQUÉ (ne peut pas ouvrir)")
    print()
    print("Causes possibles:")
    print("  1. FIX 8 (Over-Trading Protection) bloque les trades")
    print("  2. daily_loss_limit_reached = True")
    print("  3. Bug dans _open_position()")
    print("  4. Condition RSI jamais satisfaite")
    print()
    print("RECOMMANDATION: Vérifier le code de _open_position()")

elif close_rate < 30:
    print("HYPOTHÈSE PRINCIPALE: L'agent OUVRE mais ne FERME pas")
    print()
    print("Causes possibles:")
    print("  1. TP/SL trop loin (ATR trop grand)")
    print("  2. Bug dans _close_position()")
    print("  3. Check TP/SL pas exécuté dans _update_state()")
    print()
    print("RECOMMANDATION: Réduire le multiplicateur ATR pour SL")

elif avg_hold > avg_trade:
    print("HYPOTHÈSE PRINCIPALE: L'agent a APPRIS que HOLD est meilleur")
    print()
    print("Causes possibles:")
    print("  1. Reward shaping favorise l'inaction")
    print("  2. Pénalités trop fortes pour les pertes")
    print("  3. Pas assez de reward pour les trades gagnants")
    print()
    print("RECOMMANDATION: Augmenter le reward pour les trades (+5.0)")

else:
    print("HYPOTHÈSE: Agent fonctionne mais avec actions random")
    print()
    print("RECOMMANDATION: Tester avec un modèle entraîné pour voir")
    print("               le vrai comportement appris.")

print()
print("=" * 80)
print("ANALYSE COMPORTEMENTALE TERMINÉE")
print("=" * 80)
