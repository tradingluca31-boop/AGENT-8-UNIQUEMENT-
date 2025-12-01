r"""
INTERVIEW TRADES - DIAGNOSTIC COMPLET
================================================================================
POURQUOI 0 TRADES ?

Ce script pose les questions critiques:
  Q1: Les positions S'OUVRENT-elles ?
  Q2: Si oui, POURQUOI ne se ferment-elles pas ?
  Q3: Si non, QU'EST-CE QUI BLOQUE l'ouverture ?
  Q4: Le TP/SL est-il ATTEIGNABLE ?
  Q5: L'ATR est-il CORRECT ?

Duration: ~3 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import Counter
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
print("INTERVIEW TRADES - DIAGNOSTIC COMPLET")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Import
print("[1/3] Importing...")
from trading_env import GoldTradingEnvAgent8
from data_loader import load_data_agent8
from feature_engineering import calculate_all_features
print("       [OK]\n")

# Load data
print("[2/3] Loading data...")
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
print(f"       [PRICES] Close: ${prices_df['close'].iloc[1000]:.2f}")
print(f"       [OK] Data: {features_df.shape}\n")

# Create environment with VERBOSE
print("[3/3] Creating environment (VERBOSE=True)...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_episode_steps=5_000,
    verbose=True,
    training_mode=True
)
print(f"       [OK]\n")

# ================================================================================
print("=" * 80)
print("Q1: LES POSITIONS S'OUVRENT-ELLES ?")
print("=" * 80)
print()

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

print("Test: 5 actions BUY consecutives")
print("-" * 60)

positions_opened = 0
for i in range(5):
    obs, reward, done, truncated, info = env.step(2)  # BUY
    if env.position_opened_this_step:
        positions_opened += 1
    print(f"  Step {i+1}: action=BUY | position={env.position_side} | opened_this_step={env.position_opened_this_step} | reward={reward:.2f}")

print()
if positions_opened > 0:
    print(f"[OK] {positions_opened} position(s) OUVERTE(S)")
else:
    print("[X] AUCUNE POSITION OUVERTE!")
    print()
    print("    CAUSES POSSIBLES:")
    print("    1. FIX 8 (Over-Trading): Bloque si < 10 bars depuis dernier trade")
    print("    2. daily_loss_limit_reached = True")
    print("    3. position_side != 0 (deja en position)")
print()

# ================================================================================
print("=" * 80)
print("Q2: LE TP/SL EST-IL ATTEIGNABLE ?")
print("=" * 80)
print()

# Reset and open a fresh position
obs, _ = env.reset()
env.set_global_timestep(50000)

# Force open by trying multiple times
for _ in range(20):
    obs, reward, done, truncated, info = env.step(2)
    if env.position_side != 0:
        break

if env.position_side != 0:
    entry = env.entry_price
    sl = env.stop_loss_price
    tp = env.take_profit_price
    current = prices_df['close'].iloc[env.current_step]

    print(f"Position ouverte!")
    print(f"  Entry Price:       ${entry:.2f}")
    print(f"  Current Price:     ${current:.2f}")
    print(f"  Stop Loss:         ${sl:.2f}")
    print(f"  Take Profit:       ${tp:.2f}")
    print()

    # Calculate distances
    sl_distance = abs(entry - sl)
    tp_distance = abs(tp - entry)

    print(f"  Distance to SL:    ${sl_distance:.2f} ({sl_distance/entry*100:.2f}%)")
    print(f"  Distance to TP:    ${tp_distance:.2f} ({tp_distance/entry*100:.2f}%)")
    print()

    # Check ATR
    if hasattr(env, '_get_atr'):
        atr = env._get_atr()
        print(f"  ATR utilise:       ${atr:.2f}")
        print(f"  ATR % du prix:     {atr/entry*100:.2f}%")
        print()

        # Is TP reachable in reasonable time?
        # Gold moves ~1-2% per day on average
        daily_move_pct = 1.5  # %
        daily_move_usd = entry * daily_move_pct / 100

        days_to_tp = tp_distance / daily_move_usd
        days_to_sl = sl_distance / daily_move_usd

        print(f"  Mouvement daily moyen:  ${daily_move_usd:.2f} (~{daily_move_pct}%)")
        print(f"  Jours estimes pour TP:  {days_to_tp:.1f} jours")
        print(f"  Jours estimes pour SL:  {days_to_sl:.1f} jours")
        print()

        if days_to_tp > 30:
            print("[X] TP TROP LOIN! Plus de 30 jours pour l'atteindre")
            print("    -> ATR probablement trop grand")
        elif days_to_sl > 10:
            print("[WARNING] SL assez loin ({days_to_sl:.1f} jours)")
        else:
            print("[OK] TP/SL semblent atteignables")
else:
    print("[X] Impossible d'ouvrir une position pour tester TP/SL")

print()

# ================================================================================
print("=" * 80)
print("Q3: SIMULATION - QUE SE PASSE-T-IL SUR 100 STEPS ?")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

positions_opened_total = 0
positions_closed_total = 0
actions_taken = {0: 0, 1: 0, 2: 0}

print("Simulation de 100 steps avec actions aleatoires...")
print("-" * 60)

for i in range(100):
    # Random action
    action = np.random.choice([0, 1, 2])
    actions_taken[action] += 1

    obs, reward, done, truncated, info = env.step(action)

    if env.position_opened_this_step:
        positions_opened_total += 1
        print(f"  Step {i+1}: [OPEN] Position ouverte! (action={['SELL','HOLD','BUY'][action]})")

    if env.position_closed_this_step:
        positions_closed_total += 1
        print(f"  Step {i+1}: [CLOSE] Position fermee! PnL=${env.last_closed_pnl:.2f}")

    if done or truncated:
        print(f"  Step {i+1}: Episode termine")
        break

print()
print(f"Resume 100 steps:")
print(f"  Actions: SELL={actions_taken[0]}, HOLD={actions_taken[1]}, BUY={actions_taken[2]}")
print(f"  Positions ouvertes:  {positions_opened_total}")
print(f"  Positions fermees:   {positions_closed_total}")
print(f"  Trades (closed):     {len(env.trades)}")
print()

# ================================================================================
print("=" * 80)
print("Q4: VERIFICATION FIX 8 (Over-Trading Protection)")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

print("FIX 8 bloque les trades si < 10 bars depuis le dernier trade")
print()
print(f"  last_trade_open_step: {env.last_trade_open_step}")
print(f"  current_step:         {env.current_step}")
print(f"  Difference:           {env.current_step - env.last_trade_open_step} bars")
print()

if env.current_step - env.last_trade_open_step < 10:
    print("[X] FIX 8 BLOQUE! Moins de 10 bars depuis le dernier trade")
    print("    -> C'est normal au debut, mais devrait se debloquer")
else:
    print("[OK] FIX 8 ne bloque pas (>= 10 bars)")

print()

# ================================================================================
print("=" * 80)
print("Q5: SIMULATION LONGUE - 500 STEPS")
print("=" * 80)
print()

obs, _ = env.reset()
env.set_global_timestep(50000)

opened = 0
closed = 0

print("Simulation de 500 steps...")

for i in range(500):
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, info = env.step(action)

    if env.position_opened_this_step:
        opened += 1
    if env.position_closed_this_step:
        closed += 1

    if done or truncated:
        obs, _ = env.reset()

print(f"  Positions ouvertes:  {opened}")
print(f"  Positions fermees:   {closed}")
print(f"  Trades (closed):     {len(env.trades)}")
print()

# ================================================================================
print("=" * 80)
print("DIAGNOSTIC FINAL")
print("=" * 80)
print()

if opened == 0:
    print("[X] PROBLEME: Aucune position ouverte sur 500 steps!")
    print()
    print("CAUSES POSSIBLES:")
    print("  1. FIX 8 trop agressif (10 bars entre trades)")
    print("  2. daily_loss_limit_reached bloque")
    print("  3. Bug dans _open_position()")
    print()
    print("SOLUTION:")
    print("  -> Reduire FIX 8 de 10 bars a 5 bars")
    print("  -> Ou desactiver temporairement FIX 8")

elif closed == 0:
    print("[X] PROBLEME: Positions ouvertes mais JAMAIS fermees!")
    print()
    print("CAUSES POSSIBLES:")
    print("  1. TP/SL trop loin (ATR trop grand)")
    print("  2. Bug dans _close_position()")
    print("  3. Bug dans _update_state() (check TP/SL)")
    print()
    print("SOLUTION:")
    print("  -> Verifier le calcul de l'ATR")
    print("  -> Reduire le multiplicateur ATR pour SL")

else:
    print(f"[OK] {opened} positions ouvertes, {closed} fermees")
    print()
    print("L'environnement fonctionne!")
    print("Si le smoke test montre 0 trades, c'est peut-etre:")
    print("  1. Le PPO n'explore pas assez (entropy trop basse)")
    print("  2. Pas assez de steps (10K trop court)")

print()
print("=" * 80)
print("INTERVIEW TERMINE")
print("=" * 80)
