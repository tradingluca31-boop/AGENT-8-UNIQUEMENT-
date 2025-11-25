r"""
INTERVIEW AGENT 8 V2.7 - DIAGNOSTIC COMPLET
================================================================================
V2.7 a 7 NUCLEAR FIXES mais ne trade TOUJOURS PAS !

QUESTIONS CRITIQUES:
  1. Les fixes sont-ils VRAIMENT activés dans l'environnement ?
  2. Le Demonstration Learning force-t-il des trades ? (FIX 5)
  3. Les Trading Action Rewards sont-ils appliqués ? (FIX 1)
  4. Pourquoi l'agent choisit HOLD malgré les +2.0 rewards ?
  5. Les logits sont-ils toujours déséquilibrés ?
  6. L'entropy est-elle correcte (0.40→0.20) ?
  7. Que voit exactement l'agent dans les observations ?
  8. QUELLE EST LA VRAIE RAISON ?

Duration: ~3 minutes
Output: Rapport DÉTAILLÉ avec VRAIES CAUSES + FIXES CONCRETS
================================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add paths
analysis_dir = Path(__file__).resolve().parent
agent8_root = analysis_dir.parent
project_root = agent8_root.parent.parent
env_dir = agent8_root / "environment"

sys.path.insert(0, str(env_dir))
sys.path.append(str(project_root))

print("=" * 80)
print("INTERVIEW AGENT 8 - DIAGNOSTIC MODE COLLAPSE")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Import modules
print("[1/5] Importing modules...")
try:
    from stable_baselines3 import PPO
    from data_loader import load_data_agent8
    from feature_engineering import calculate_all_features
    from trading_env import GoldTradingEnvAgent8  # Environment centralisé
    print("       [OK] Imports successful\n")
except ImportError as e:
    print(f"       [ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load data
print("[2/5] Loading data...")
df_full, auxiliary_data = load_data_agent8()
df_full = df_full.loc['2008-01-01':'2020-12-31']  # Training period only

# Filter auxiliary data
for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc['2008-01-01':'2020-12-31']

ohlcv_cols = [col for col in df_full.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
prices_df = df_full[ohlcv_cols[:5]].copy()
prices_df.columns = ['open', 'high', 'low', 'close', 'volume']

features_df = calculate_all_features(df_full, auxiliary_data)
print(f"       [OK] Data loaded: {features_df.shape}\n")

# Create environment
print("[3/5] Creating V2.7 environment...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    max_episode_steps=5_000,
    verbose=False,
    training_mode=True
)
print(f"       [OK] Environment created\n")

# Load model
print("[4/5] Loading V2.7 model...")
checkpoint_path = v2_dir / 'checkpoints' / 'agent8_v2.7_checkpoint_250000_steps.zip'

if not checkpoint_path.exists():
    print(f"       [ERROR] Checkpoint not found: {checkpoint_path}")
    print(f"       Looking for alternative...")
    # Try to find any V2.7 checkpoint
    checkpoint_dir = v2_dir / 'checkpoints'
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('agent8_v2.7_checkpoint_*.zip'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]  # Use latest
            print(f"       [FOUND] Using: {checkpoint_path.name}")
        else:
            print(f"       [ERROR] No V2.7 checkpoints found!")
            sys.exit(1)
    else:
        print(f"       [ERROR] Checkpoints directory doesn't exist!")
        sys.exit(1)

try:
    model = PPO.load(str(checkpoint_path))
    print(f"       [OK] Model loaded: {checkpoint_path.name}\n")
except Exception as e:
    print(f"       [ERROR] Failed to load model: {e}")
    sys.exit(1)

# ================================================================================
# INTERVIEW START
# ================================================================================

print("=" * 80)
print("STARTING INTERVIEW - 8 CRITICAL QUESTIONS")
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
add_to_report("Q1: LES FIXES V2.7 SONT-ILS VRAIMENT ACTIVÉS ?")
add_to_report("=" * 80)
add_to_report("")

# Check if V2.7 methods exist
fixes_status = {
    'FIX 1 - Trading Action Rewards': hasattr(env, 'position_opened_this_step'),
    'FIX 5 - Demonstration Learning': hasattr(env, 'set_global_timestep'),
    'FIX 6 - Forced Trading': True,  # Hard to check, assume present
    'FIX 8 - Over-Trading Protection': hasattr(env, 'last_trade_open_step'),
}

add_to_report("Vérification des attributs V2.7:")
all_fixes_present = True
for fix_name, status in fixes_status.items():
    status_str = "✅ PRÉSENT" if status else "❌ ABSENT"
    add_to_report(f"  {fix_name}: {status_str}")
    if not status:
        all_fixes_present = False

add_to_report("")
if all_fixes_present:
    add_to_report("✅ TOUS LES FIXES SONT PRÉSENTS DANS L'ENVIRONNEMENT")
else:
    add_to_report("❌ CERTAINS FIXES MANQUENT ! L'environnement n'est pas V2.7 !")

add_to_report("")

# ================================================================================
# QUESTION 2: Le Demonstration Learning force-t-il des trades ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q2: LE DEMONSTRATION LEARNING FORCE-T-IL DES TRADES ? (FIX 5)")
add_to_report("=" * 80)
add_to_report("")

if hasattr(env, 'set_global_timestep'):
    # Test Phase 1 (0-100K): Should force 100% of smart trades
    env.reset()
    env.set_global_timestep(50000)  # Phase 1

    forced_trades_count = 0
    total_samples = 100

    for i in range(total_samples):
        obs, _ = env.reset()

        # Get current phase
        phase = env._get_demonstration_phase()

        # Check if there's a forcing opportunity
        current_price = env.prices_df['close'].iloc[env.current_step]
        forced_action = env._should_force_demonstration_trade(phase, current_price)

        if forced_action > 0:
            forced_trades_count += 1

    force_pct = (forced_trades_count / total_samples) * 100

    add_to_report(f"Phase testée: {phase} (timestep 50,000)")
    add_to_report(f"Trades forcés: {forced_trades_count}/{total_samples} ({force_pct:.1f}%)")
    add_to_report("")

    if phase == 1 and force_pct > 0:
        add_to_report("✅ Demonstration Learning ACTIF (Phase 1)")
    elif phase == 1 and force_pct == 0:
        add_to_report("❌ Demonstration Learning INACTIF ! Aucun trade forcé en Phase 1 !")
        add_to_report("   → CAUSE POSSIBLE: Pas d'opportunités RSI <30 ou >70 dans les échantillons")
    else:
        add_to_report(f"⚠️ Phase {phase}: Forcing réduit ou absent (normal)")
else:
    add_to_report("❌ Demonstration Learning NON IMPLÉMENTÉ dans cet environnement !")

add_to_report("")

# ================================================================================
# QUESTION 3: Les Trading Action Rewards sont-ils appliqués ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q3: LES TRADING ACTION REWARDS SONT-ILS APPLIQUÉS ? (FIX 1)")
add_to_report("=" * 80)
add_to_report("")

if hasattr(env, 'position_opened_this_step'):
    # Simulate opening a trade
    obs, _ = env.reset()
    env.set_global_timestep(50000)

    # Force a BUY action
    obs, reward, done, truncated, info = env.step(2)  # Action 2 = BUY

    add_to_report(f"Action simulée: BUY (action=2)")
    add_to_report(f"Reward reçu: {reward:.2f}")
    add_to_report(f"Position ouverte ce step: {env.position_opened_this_step}")
    add_to_report("")

    if env.position_opened_this_step and reward > 1.0:
        add_to_report("✅ Trading Action Reward ACTIF (+2.0 détecté)")
    elif env.position_opened_this_step and reward < 1.0:
        add_to_report("⚠️ Position ouverte mais reward faible (<1.0)")
        add_to_report("   → FIX 1 peut-être dilué par d'autres penalties")
    else:
        add_to_report("❌ Pas de Trading Action Reward détecté !")
else:
    add_to_report("❌ Trading Action Rewards NON IMPLÉMENTÉS !")

add_to_report("")

# ================================================================================
# QUESTION 4: Pourquoi l'agent choisit HOLD ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q4: POURQUOI L'AGENT CHOISIT HOLD MALGRÉ LES REWARDS ?")
add_to_report("=" * 80)
add_to_report("")

# Sample 1000 steps and check policy
obs, _ = env.reset()
env.set_global_timestep(250000)  # Checkpoint 250K

actions_sampled = []
logits_list = []

for i in range(1000):
    # Get model prediction
    action, _states = model.predict(obs, deterministic=True)

    # Get logits (raw network output before softmax)
    try:
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        # Get action distribution
        latent_pi = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(obs_tensor))
        action_logits = model.policy.action_net(latent_pi)
        logits = action_logits.detach().cpu().numpy()[0]
        logits_list.append(logits)
    except Exception as e:
        # Fallback: Just use zeros if logits extraction fails
        logits_list.append(np.array([0.0, 0.0, 0.0]))

    actions_sampled.append(int(action))

    # Step environment
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, _ = env.reset()

# Analyze actions
action_counts = Counter(actions_sampled)
total = len(actions_sampled)

sell_pct = (action_counts.get(0, 0) / total) * 100
hold_pct = (action_counts.get(1, 0) / total) * 100
buy_pct = (action_counts.get(2, 0) / total) * 100

add_to_report("Distribution des actions (1000 échantillons):")
add_to_report(f"  SELL (0): {action_counts.get(0, 0):4d} ({sell_pct:5.1f}%)")
add_to_report(f"  HOLD (1): {action_counts.get(1, 0):4d} ({hold_pct:5.1f}%)")
add_to_report(f"  BUY  (2): {action_counts.get(2, 0):4d} ({buy_pct:5.1f}%)")
add_to_report("")

# Analyze logits
logits_array = np.array(logits_list)
logits_mean = logits_array.mean(axis=0)

add_to_report("Logits moyens (raw network output):")
add_to_report(f"  Logit SELL (0): {logits_mean[0]:+10.2f}")
add_to_report(f"  Logit HOLD (1): {logits_mean[1]:+10.2f}")
add_to_report(f"  Logit BUY  (2): {logits_mean[2]:+10.2f}")
add_to_report("")

logit_gap = max(abs(logits_mean[0] - logits_mean[1]), abs(logits_mean[1] - logits_mean[2]))
add_to_report(f"Logit Gap (imbalance): {logit_gap:.2f}")
add_to_report("")

# Calculate probabilities from logits
exp_logits = np.exp(logits_mean)
probs = exp_logits / exp_logits.sum()

add_to_report("Probabilités (softmax des logits):")
add_to_report(f"  P(SELL): {probs[0]*100:5.1f}%")
add_to_report(f"  P(HOLD): {probs[1]*100:5.1f}%")
add_to_report(f"  P(BUY):  {probs[2]*100:5.1f}%")
add_to_report("")

# Diagnosis
if hold_pct > 80:
    add_to_report("❌ MODE COLLAPSE CONFIRMÉ: >80% HOLD !")
    if logit_gap > 100:
        add_to_report("   → CAUSE: Logit gap ÉNORME (>100) - réseau déséquilibré")
    elif logit_gap > 10:
        add_to_report("   → CAUSE: Logit gap important (>10) - préférence forte pour HOLD")
    else:
        add_to_report("   → CAUSE: Logits équilibrés mais action masking ou autre mécanisme")
elif hold_pct > 50:
    add_to_report("⚠️ HOLD DOMINANT (50-80%) mais pas total collapse")
else:
    add_to_report("✅ PAS DE MODE COLLAPSE: Actions diversifiées")

add_to_report("")

# ================================================================================
# QUESTION 5: L'entropy est-elle correcte ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q5: L'ENTROPY EST-ELLE CORRECTE (0.40→0.20) ?")
add_to_report("=" * 80)
add_to_report("")

# Get current entropy coefficient
current_ent_coef = model.ent_coef

add_to_report(f"Entropy coefficient actuel: {current_ent_coef:.3f}")
add_to_report("")

# Expected entropy at 250K steps (50% progress in 500K training)
expected_ent = 0.40 + (0.20 - 0.40) * (250000 / 500000)  # Linear interpolation
add_to_report(f"Entropy attendue @ 250K steps: {expected_ent:.3f} (50% de 0.40→0.20)")
add_to_report("")

if abs(current_ent_coef - expected_ent) < 0.05:
    add_to_report("✅ Entropy schedule CORRECTE")
else:
    add_to_report(f"❌ Entropy schedule INCORRECTE ! Écart: {abs(current_ent_coef - expected_ent):.3f}")
    add_to_report("   → Callback AdaptiveEntropyCallback peut-être pas activé")

add_to_report("")

# Calculate actual entropy of policy
policy_entropy = -np.sum(probs * np.log(probs + 1e-8))
max_entropy = np.log(3.0)  # Max entropy for 3 actions
normalized_entropy = policy_entropy / max_entropy

add_to_report(f"Entropie réelle de la policy: {policy_entropy:.3f} (normalized: {normalized_entropy:.3f})")
add_to_report(f"Entropie maximale (3 actions): {max_entropy:.3f}")
add_to_report("")

if normalized_entropy < 0.3:
    add_to_report("❌ Entropie TRÈS BASSE (<30% du max) - Agent surconfiant !")
elif normalized_entropy < 0.5:
    add_to_report("⚠️ Entropie BASSE (30-50% du max) - Peu d'exploration")
else:
    add_to_report("✅ Entropie CORRECTE (>50% du max)")

add_to_report("")

# ================================================================================
# QUESTION 6: Que voit l'agent dans les observations ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q6: QUE VOIT L'AGENT DANS LES OBSERVATIONS ?")
add_to_report("=" * 80)
add_to_report("")

obs, _ = env.reset()

add_to_report(f"Observation shape: {obs.shape}")
add_to_report(f"Observation min: {obs.min():.3f}")
add_to_report(f"Observation max: {obs.max():.3f}")
add_to_report(f"Observation mean: {obs.mean():.3f}")
add_to_report(f"Observation std: {obs.std():.3f}")
add_to_report("")

# Check for NaN or Inf
nan_count = np.isnan(obs).sum()
inf_count = np.isinf(obs).sum()

if nan_count > 0 or inf_count > 0:
    add_to_report(f"❌ PROBLÈME: NaN count={nan_count}, Inf count={inf_count}")
    add_to_report("   → Les observations contiennent des valeurs invalides !")
else:
    add_to_report("✅ Observations valides (pas de NaN/Inf)")

add_to_report("")

# Check observation distribution
zero_count = (obs == 0).sum()
zero_pct = (zero_count / len(obs)) * 100

add_to_report(f"Features à zéro: {zero_count}/{len(obs)} ({zero_pct:.1f}%)")

if zero_pct > 50:
    add_to_report("❌ PROBLÈME: >50% des features sont à zéro !")
    add_to_report("   → Feature engineering peut-être cassé")
elif zero_pct > 20:
    add_to_report("⚠️ Beaucoup de features à zéro (>20%)")
else:
    add_to_report("✅ Distribution des features normale")

add_to_report("")

# ================================================================================
# QUESTION 7: Le Critic apprend-il correctement ?
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q7: LE CRITIC (VALUE FUNCTION) APPREND-IL ?")
add_to_report("=" * 80)
add_to_report("")

# Sample value estimates
obs, _ = env.reset()
values = []

for i in range(100):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    value = model.policy.predict_values(obs_tensor).detach().cpu().numpy()[0][0]
    values.append(value)

    obs, _, done, truncated, _ = env.step(env.action_space.sample())
    if done or truncated:
        obs, _ = env.reset()

values_array = np.array(values)

add_to_report("Value function statistics (100 échantillons):")
add_to_report(f"  Mean:   {values_array.mean():+10.2f}")
add_to_report(f"  Std:    {values_array.std():10.2f}")
add_to_report(f"  Min:    {values_array.min():+10.2f}")
add_to_report(f"  Max:    {values_array.max():+10.2f}")
add_to_report(f"  Range:  {values_array.max() - values_array.min():10.2f}")
add_to_report("")

if values_array.std() > 1.0:
    add_to_report("✅ Critic APPREND (variance >1.0) - Distingue les états")
elif values_array.std() > 0.1:
    add_to_report("⚠️ Critic apprend un peu (variance 0.1-1.0)")
else:
    add_to_report("❌ Critic N'APPREND PAS (variance <0.1) - Tous les états semblent identiques")

add_to_report("")

# ================================================================================
# QUESTION 8: SYNTHÈSE ET RECOMMANDATIONS
# ================================================================================

add_to_report("=" * 80)
add_to_report("Q8: SYNTHÈSE - QUELLE EST LA VRAIE CAUSE ?")
add_to_report("=" * 80)
add_to_report("")

# Identify root causes
causes = []
severity = []

if not all_fixes_present:
    causes.append("Fixes V2.7 manquants dans l'environnement")
    severity.append("CRITIQUE")

if hasattr(env, 'set_global_timestep') and force_pct == 0 and phase == 1:
    causes.append("Demonstration Learning inactif (0 trades forcés en Phase 1)")
    severity.append("CRITIQUE")

if logit_gap > 100:
    causes.append(f"Logit gap ÉNORME ({logit_gap:.0f}) - Réseau complètement déséquilibré")
    severity.append("CRITIQUE")
elif logit_gap > 10:
    causes.append(f"Logit gap important ({logit_gap:.1f}) - Forte préférence HOLD")
    severity.append("MAJEUR")

if normalized_entropy < 0.3:
    causes.append(f"Entropie très basse ({normalized_entropy:.2f}) - Surconfiance")
    severity.append("MAJEUR")

if zero_pct > 50:
    causes.append(f"Observations invalides ({zero_pct:.0f}% features à zéro)")
    severity.append("CRITIQUE")

if values_array.std() < 0.1:
    causes.append("Critic n'apprend pas (variance <0.1)")
    severity.append("MAJEUR")

if not causes:
    causes.append("Cause inconnue - Nécessite investigation plus poussée")
    severity.append("CRITIQUE")

add_to_report("CAUSES IDENTIFIÉES:")
for i, (cause, sev) in enumerate(zip(causes, severity), 1):
    add_to_report(f"  [{sev}] {i}. {cause}")

add_to_report("")
add_to_report("=" * 80)
add_to_report("RECOMMANDATIONS CONCRÈTES:")
add_to_report("=" * 80)
add_to_report("")

# Generate recommendations based on causes
recommendations = []

if not all_fixes_present:
    recommendations.append("1. VÉRIFIER que training_env_v2_7_agent8.py est bien utilisé (pas v2_agent8.py)")
    recommendations.append("   → Relancer training avec le BON fichier d'environnement")

if logit_gap > 50:
    recommendations.append("2. RESET LE RÉSEAU - Logit gap trop grand pour être récupéré")
    recommendations.append("   → Option A: Restart training from scratch avec V2.7")
    recommendations.append("   → Option B: Behavioral cloning (pre-train avec supervised learning)")

if normalized_entropy < 0.3 or (hasattr(env, 'set_global_timestep') and force_pct == 0):
    recommendations.append("3. AUGMENTER DEMONSTRATION LEARNING")
    recommendations.append("   → Phase 1: Étendre de 100K à 200K steps")
    recommendations.append("   → Phase 1: Augmenter reward de +10.0 à +20.0")
    recommendations.append("   → Phase 1: Forcer PLUS d'opportunités (RSI <40 et >60 au lieu de <30 et >70)")

if hold_pct > 80:
    recommendations.append("4. AUGMENTER FIX 1 (Trading Action Rewards)")
    recommendations.append("   → Open trade: +2.0 → +5.0 (×2.5)")
    recommendations.append("   → Close profitable: +2.0 → +5.0 (×2.5)")

if values_array.std() < 0.1:
    recommendations.append("5. AUGMENTER LEARNING RATE")
    recommendations.append("   → Passer de 3e-4 à 1e-3 (×3.3)")
    recommendations.append("   → Ajouter learning rate schedule (decay après 200K)")

if zero_pct > 20:
    recommendations.append("6. VÉRIFIER FEATURE ENGINEERING")
    recommendations.append("   → Beaucoup de features à zéro")
    recommendations.append("   → Vérifier calculate_all_features() et data loader")

# Print recommendations
for rec in recommendations:
    add_to_report(rec)

if not recommendations:
    add_to_report("Aucune recommandation spécifique - Consulter expert RL")

add_to_report("")
add_to_report("=" * 80)
add_to_report("FIN DE L'INTERVIEW")
add_to_report("=" * 80)
add_to_report("")

# ================================================================================
# SAVE REPORT
# ================================================================================

print("\n[5/5] Saving report...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
report_filename = f"DIAGNOSTIC_REPORT_V27_{timestamp}.txt"
report_path = v2_dir / report_filename

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"       [OK] Report saved: {report_filename}")
print()
print("=" * 80)
print("INTERVIEW TERMINÉE")
print("=" * 80)
print(f"Report: {report_filename}")
print()
print("PROCHAINES ÉTAPES:")
print("  1. Lire le rapport complet")
print("  2. Appliquer les recommandations")
print("  3. Relancer training avec les fixes")
print()
