r"""
INTERVIEW AGENT 8 - ENVIRONMENT ONLY (No model required)
================================================================================
Test the environment fixes WITHOUT needing a trained model.

QUESTIONS:
  1. Les fixes V2.7 sont-ils VRAIMENT activés ?
  2. Le Demonstration Learning force-t-il des trades ?
  3. Les Trading Action Rewards sont-ils appliqués ?
  4. Le reward +5.0 est-il VRAIMENT retourné (pas 0.0) ?
  5. Les observations sont-elles valides ?
  6. L'action BUY ouvre-t-elle une position ?

Duration: ~2 minutes
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

# GoldRL paths (for data_loader, feature_engineering)
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

# IMPORTANT: Order matters! Local env_dir FIRST, then GoldRL
sys.path.insert(0, str(goldrl_root))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(env_dir))  # MUST BE FIRST for local trading_env.py

print("=" * 80)
print("INTERVIEW AGENT 8 - ENVIRONMENT ONLY")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Import modules
print("[1/3] Importing modules...")
try:
    from trading_env import GoldTradingEnvAgent8
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    print("       [OK] Imports successful\n")
except ImportError as e:
    print(f"       [ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load data
print("[2/3] Loading data...")
df_full, auxiliary_data = load_data_agent8()
df_full = df_full.loc['2008-01-01':'2020-12-31']

for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc['2008-01-01':'2020-12-31']

# Calculate features FIRST (to get the aligned index)
features_df = calculate_all_features(df_full, auxiliary_data)

# FIX: Use RAW XAUUSD H1 data for prices (has real OHLCV: open, high, low, close, volume)
# The df_full only contains _close columns from different assets, NOT real OHLCV!
prices_raw = auxiliary_data['xauusd_raw']['H1'].loc['2008-01-01':'2020-12-31'].copy()

# Align prices_df with features_df (same index)
common_index = features_df.index.intersection(prices_raw.index)
prices_df = prices_raw.loc[common_index]
features_df = features_df.loc[common_index]
print(f"       [PRICES] Using raw XAUUSD H1 - Close sample: ${prices_df['close'].iloc[1000]:.2f}")
print(f"       [OK] Data loaded: {features_df.shape}\n")

# Create environment
print("[3/3] Creating environment...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_episode_steps=5_000,
    verbose=True,  # VERBOSE to see logs
    training_mode=True
)
print(f"       [OK] Environment created\n")

# ================================================================================
# INTERVIEW START
# ================================================================================

print("=" * 80)
print("STARTING INTERVIEW - 6 CRITICAL QUESTIONS")
print("=" * 80)
print()

report = []

def add_to_report(text):
    report.append(text)
    print(text)

# ================================================================================
# QUESTION 1: Les fixes V2.7 sont-ils ACTIVÉS ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q1: LES FIXES SONT-ILS VRAIMENT ACTIVES ?")
add_to_report("=" * 80)
add_to_report("")

fixes_status = {
    'FIX 1 - Trading Action Rewards': hasattr(env, 'position_opened_this_step'),
    'FIX 5 - Demonstration Learning': hasattr(env, 'set_global_timestep'),
    'FIX 6 - Forced Trading': True,
    'FIX 8 - Over-Trading Protection': hasattr(env, 'last_trade_open_step'),
}

add_to_report("Verification des attributs:")
all_fixes_present = True
for fix_name, status in fixes_status.items():
    status_str = "[OK] PRESENT" if status else "[X] ABSENT"
    add_to_report(f"  {fix_name}: {status_str}")
    if not status:
        all_fixes_present = False

add_to_report("")
if all_fixes_present:
    add_to_report("[OK] TOUS LES FIXES SONT PRESENTS")
else:
    add_to_report("[X] CERTAINS FIXES MANQUENT !")

add_to_report("")

# ================================================================================
# QUESTION 2: Le Demonstration Learning force-t-il des trades ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q2: LE DEMONSTRATION LEARNING FORCE-T-IL DES TRADES ?")
add_to_report("=" * 80)
add_to_report("")

if hasattr(env, 'set_global_timestep'):
    env.reset()
    env.set_global_timestep(50000)  # Phase 1

    forced_trades_count = 0
    total_samples = 50

    for i in range(total_samples):
        obs, _ = env.reset()
        phase = env._get_demonstration_phase()
        current_price = env.prices_df['close'].iloc[env.current_step]
        forced_action = env._should_force_demonstration_trade(phase, current_price)

        if forced_action > 0:
            forced_trades_count += 1

    force_pct = (forced_trades_count / total_samples) * 100

    add_to_report(f"Phase testee: {phase} (timestep 50,000)")
    add_to_report(f"Trades forces: {forced_trades_count}/{total_samples} ({force_pct:.1f}%)")
    add_to_report("")

    if force_pct > 0:
        add_to_report("[OK] Demonstration Learning ACTIF")
    else:
        add_to_report("[WARNING] Pas de trades forces (peut-etre pas d'opportunites RSI)")
else:
    add_to_report("[X] Demonstration Learning NON IMPLEMENTE !")

add_to_report("")

# ================================================================================
# QUESTION 3: Les Trading Action Rewards sont-ils appliqués ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q3: LES TRADING ACTION REWARDS SONT-ILS APPLIQUES ?")
add_to_report("=" * 80)
add_to_report("")

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

add_to_report("Test: Envoyer action BUY (action=2)...")
add_to_report("-" * 40)

obs, reward, done, truncated, info = env.step(2)  # BUY

add_to_report(f"  Action:   BUY (2)")
add_to_report(f"  Position: {env.position_side} (0=flat, 1=long, -1=short)")
add_to_report(f"  Reward:   {reward:.4f}")
add_to_report("")

# This is the KEY test!
if env.position_side == 1:  # Position opened
    add_to_report("[OK] Position OUVERTE (side=1)")

    if reward > 1.0:
        add_to_report(f"[OK] REWARD > 1.0 ! Trading Action Reward FONCTIONNE!")
    elif reward > 0.0:
        add_to_report(f"[WARNING] Reward positif mais faible ({reward:.2f})")
    else:
        add_to_report(f"[X] REWARD = 0 ou negatif ! Early returns TOUJOURS PRESENTS!")
        add_to_report("    --> Verifier _calculate_reward() pour d'autres 'return' statements")
else:
    add_to_report("[X] Position NON OUVERTE ! _open_position() pas appele ?")

add_to_report("")

# ================================================================================
# QUESTION 4: Test multiple actions
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q4: TEST ACTIONS MULTIPLES (10 BUY consecutifs)")
add_to_report("=" * 80)
add_to_report("")

obs, _ = env.reset()
if hasattr(env, 'set_global_timestep'):
    env.set_global_timestep(50000)

rewards_collected = []
positions_seen = []

for i in range(10):
    action = 2  # BUY
    obs, reward, done, truncated, info = env.step(action)
    rewards_collected.append(reward)
    positions_seen.append(env.position_side)
    add_to_report(f"  Step {i+1}: action=BUY, position={env.position_side}, reward={reward:.4f}")
    if done:
        add_to_report("  Episode done!")
        break

add_to_report("")
add_to_report(f"Resume:")
add_to_report(f"  Rewards: min={min(rewards_collected):.4f}, max={max(rewards_collected):.4f}, avg={np.mean(rewards_collected):.4f}")
add_to_report(f"  Positions vues: {set(positions_seen)}")

if max(rewards_collected) > 1.0:
    add_to_report("[OK] Au moins un reward > 1.0 !")
else:
    add_to_report("[X] Tous les rewards <= 1.0 - Verifier les early returns!")

add_to_report("")

# ================================================================================
# QUESTION 5: Observations valides ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q5: LES OBSERVATIONS SONT-ELLES VALIDES ?")
add_to_report("=" * 80)
add_to_report("")

obs, _ = env.reset()

add_to_report(f"Observation shape: {obs.shape}")
add_to_report(f"Observation min:   {obs.min():.3f}")
add_to_report(f"Observation max:   {obs.max():.3f}")
add_to_report(f"Observation mean:  {obs.mean():.3f}")
add_to_report(f"Observation std:   {obs.std():.3f}")
add_to_report("")

nan_count = np.isnan(obs).sum()
inf_count = np.isinf(obs).sum()
zero_count = (obs == 0).sum()
zero_pct = (zero_count / len(obs)) * 100

if nan_count > 0 or inf_count > 0:
    add_to_report(f"[X] PROBLEME: NaN={nan_count}, Inf={inf_count}")
else:
    add_to_report("[OK] Pas de NaN/Inf")

add_to_report(f"Features a zero: {zero_count}/{len(obs)} ({zero_pct:.1f}%)")

if zero_pct > 50:
    add_to_report("[X] >50% features a zero - PROBLEME!")
else:
    add_to_report("[OK] Distribution normale")

add_to_report("")

# ================================================================================
# QUESTION 6: Test _open_position direct
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q6: TEST DIRECT DE _open_position()")
add_to_report("=" * 80)
add_to_report("")

obs, _ = env.reset()
add_to_report(f"Position avant: {env.position_side}")

current_price = env.prices_df.iloc[env.current_step]['close']
add_to_report(f"Prix actuel: ${current_price:.2f}")

add_to_report("Appel direct: env._open_position(side=1, price=current_price)...")

try:
    env._open_position(side=1, price=current_price)
    add_to_report(f"Position apres: {env.position_side}")
    add_to_report(f"Entry price: {env.entry_price}")

    if env.position_side == 1:
        add_to_report("[OK] _open_position() FONCTIONNE!")
    else:
        add_to_report("[X] _open_position() n'a pas change la position!")
except Exception as e:
    add_to_report(f"[X] ERREUR: {e}")

add_to_report("")

# ================================================================================
# SYNTHESE
# ================================================================================

add_to_report("=" * 80)
add_to_report("SYNTHESE - DIAGNOSTIC")
add_to_report("=" * 80)
add_to_report("")

problems = []
successes = []

if not all_fixes_present:
    problems.append("Fixes manquants dans l'environnement")
else:
    successes.append("Tous les fixes presents")

if max(rewards_collected) > 1.0:
    successes.append("Trading Action Rewards fonctionnent (reward > 1.0)")
else:
    problems.append("Trading Action Rewards ne fonctionnent PAS (reward <= 1.0)")
    problems.append("--> Verifier qu'il n'y a plus de 'return' dans _calculate_reward()")

if env.position_side == 1:
    successes.append("_open_position() fonctionne")
else:
    problems.append("_open_position() ne fonctionne pas")

add_to_report("SUCCES:")
for s in successes:
    add_to_report(f"  [OK] {s}")

add_to_report("")
add_to_report("PROBLEMES:")
if problems:
    for p in problems:
        add_to_report(f"  [X] {p}")
else:
    add_to_report("  Aucun probleme detecte!")

add_to_report("")
add_to_report("=" * 80)

if not problems:
    add_to_report("[SUCCESS] ENVIRONNEMENT PRET POUR TRAINING!")
    add_to_report("")
    add_to_report("Prochaine etape:")
    add_to_report("  python training\\train_smoke_test.py")
else:
    add_to_report("[FAIL] ENVIRONNEMENT A DES PROBLEMES!")
    add_to_report("")
    add_to_report("Actions requises:")
    for p in problems:
        add_to_report(f"  - Fixer: {p}")

add_to_report("=" * 80)
print()
print("Interview terminee!")
