r"""
NORMALIZATION AUDIT - WALL STREET GRADE ANALYSIS
================================================================================
Analyse COMPLETE de la normalisation des features pour RL Trading

OBJECTIF:
  - Identifier TOUTES les features mal normalisées
  - Catégoriser par type (prix, pourcentage, ratio, etc.)
  - Proposer les transformations appropriées
  - Générer rapport institutionnel

STANDARD WALL STREET:
  - Features pour NN devraient être dans [-1, 1] ou [0, 1]
  - Outliers clippés à ±3 std
  - Pas de valeurs > 10 en absolu (sauf exceptions documentées)

Usage:
  python analysis/normalization_audit.py
  python analysis/normalization_audit.py --output report.csv
  python analysis/normalization_audit.py --fix  # Apply fixes

Duration: ~2 minutes
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from collections import defaultdict
import json
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

print("=" * 100)
print("NORMALIZATION AUDIT - WALL STREET GRADE ANALYSIS")
print("=" * 100)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
print()

# ================================================================================
# ARGUMENTS
# ================================================================================

parser = argparse.ArgumentParser(description='Normalization Audit')
parser.add_argument('--output', type=str, default=None, help='Output CSV path')
parser.add_argument('--fix', action='store_true', help='Generate fix code')
args = parser.parse_args()

# ================================================================================
# IMPORTS
# ================================================================================

print("[1/7] Importing modules...")
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

print("[2/7] Loading data...")
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
print(f"       Total features: {features_df.shape[1]}")
print()

# ================================================================================
# FEATURE CATEGORIZATION (WALL STREET STANDARD)
# ================================================================================

print("[3/7] Categorizing features...")

# Categories and expected ranges
FEATURE_CATEGORIES = {
    'price_raw': {
        'patterns': ['close', 'open', 'high', 'low', 'sma_', 'ema_', 'bb_upper', 'bb_lower', 'pivot', 'resistance', 'support', 'vpoc', 'vah', 'val'],
        'expected_range': (0, 5000),  # Gold price range
        'normalization': 'minmax_global',  # Normalize based on historical min/max
        'severity': 'CRITICAL'
    },
    'price_ratio': {
        'patterns': ['price_vs_', 'distance_to_', 'fib_distance', 'bb_width', 'returns_', 'log_returns'],
        'expected_range': (-0.5, 0.5),
        'normalization': 'clip_zscore',  # Z-score with clipping
        'severity': 'HIGH'
    },
    'percentage': {
        'patterns': ['rsi_', 'stochastic_', 'williams_r', 'mfi_', 'percentile', '_pct', 'va_position'],
        'expected_range': (0, 100),
        'normalization': 'divide_100',  # Divide by 100 to get 0-1
        'severity': 'MEDIUM'
    },
    'oscillator': {
        'patterns': ['macd', 'momentum', 'roc', 'tsi', 'cci', 'adx', 'acceleration'],
        'expected_range': (-100, 100),
        'normalization': 'zscore',
        'severity': 'HIGH'
    },
    'volume': {
        'patterns': ['volume', 'va_width'],
        'expected_range': (0, 1e9),
        'normalization': 'log_zscore',  # Log transform + Z-score
        'severity': 'CRITICAL'
    },
    'atr_volatility': {
        'patterns': ['atr_', 'historical_vol'],
        'expected_range': (0, 100),
        'normalization': 'percentile',  # Normalize to percentile
        'severity': 'HIGH'
    },
    'binary': {
        'patterns': ['is_strong', 'is_best', 'extreme', 'contrarian', 'fib_zone'],
        'expected_range': (0, 1),
        'normalization': 'none',  # Already binary
        'severity': 'OK'
    },
    'zscore': {
        'patterns': ['zscore', 'divergence'],
        'expected_range': (-4, 4),
        'normalization': 'clip',  # Clip to [-3, 3]
        'severity': 'MEDIUM'
    },
    'bias': {
        'patterns': ['bias', 'regime', 'impulse', 'strength', 'surge'],
        'expected_range': (-1, 1),
        'normalization': 'clip',
        'severity': 'MEDIUM'
    },
    'temporal': {
        'patterns': ['hour', 'day_of_week', 'session', 'weekend', 'asia', 'london', 'ny'],
        'expected_range': (0, 24),
        'normalization': 'cyclical',  # Sin/cos encoding
        'severity': 'LOW'
    },
    'cot_net': {
        'patterns': ['cot_', 'noncomm_net', 'comm_net'],
        'expected_range': (-500000, 500000),
        'normalization': 'zscore',
        'severity': 'HIGH'
    },
    'correlation': {
        'patterns': ['corr_', 'correlation'],
        'expected_range': (-1, 1),
        'normalization': 'none',  # Already normalized
        'severity': 'OK'
    },
    'macro': {
        'patterns': ['macro_score'],
        'expected_range': (-10, 10),
        'normalization': 'clip_zscore',
        'severity': 'MEDIUM'
    },
    'ratio': {
        'patterns': ['ratio', 'factor'],
        'expected_range': (0, 10),
        'normalization': 'log_clip',
        'severity': 'MEDIUM'
    },
}

def categorize_feature(feature_name):
    """Categorize a feature based on its name pattern."""
    feature_lower = feature_name.lower()

    for category, info in FEATURE_CATEGORIES.items():
        for pattern in info['patterns']:
            if pattern.lower() in feature_lower:
                return category, info

    return 'unknown', {
        'expected_range': (-100, 100),
        'normalization': 'zscore',
        'severity': 'UNKNOWN'
    }

# Categorize all features
feature_analysis = []

for col in features_df.columns:
    category, info = categorize_feature(col)

    # Get statistics
    values = features_df[col].dropna().values

    if len(values) == 0:
        continue

    stats = {
        'feature': col,
        'category': category,
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'q1': np.percentile(values, 25),
        'q99': np.percentile(values, 99),
        'zeros_pct': (values == 0).sum() / len(values) * 100,
        'nan_pct': features_df[col].isna().sum() / len(features_df) * 100,
        'expected_min': info['expected_range'][0],
        'expected_max': info['expected_range'][1],
        'normalization': info['normalization'],
        'severity': info['severity'],
    }

    # Check if in expected range
    stats['in_range'] = (stats['min'] >= stats['expected_min'] * 1.1 or stats['expected_min'] == 0) and \
                        (stats['max'] <= stats['expected_max'] * 1.1 or stats['expected_max'] == 0)

    # Check if needs normalization (Wall Street criteria)
    stats['needs_norm'] = (
        abs(stats['max']) > 10 or
        abs(stats['min']) > 10 or
        stats['std'] > 10 or
        (category == 'percentage' and stats['max'] > 1) or
        (category == 'price_raw' and stats['max'] > 100)
    )

    feature_analysis.append(stats)

print(f"       [OK] Categorized {len(feature_analysis)} features")
print()

# ================================================================================
# ANALYSIS BY CATEGORY
# ================================================================================

print("[4/7] Analyzing by category...")
print()

# Group by category
by_category = defaultdict(list)
for fa in feature_analysis:
    by_category[fa['category']].append(fa)

# Summary table
print("CATEGORY SUMMARY")
print("-" * 100)
print(f"{'Category':<20} | {'Count':>6} | {'Needs Norm':>10} | {'Severity':<10} | {'Range Example':<30}")
print("-" * 100)

critical_features = []
for category, features in sorted(by_category.items()):
    needs_norm = sum(1 for f in features if f['needs_norm'])
    severity = features[0]['severity'] if features else 'N/A'

    # Example range
    if features:
        example = features[0]
        range_str = f"[{example['min']:.2f}, {example['max']:.2f}]"
    else:
        range_str = "N/A"

    print(f"{category:<20} | {len(features):>6} | {needs_norm:>10} | {severity:<10} | {range_str:<30}")

    # Collect critical features
    for f in features:
        if f['needs_norm'] and f['severity'] in ['CRITICAL', 'HIGH']:
            critical_features.append(f)

print("-" * 100)
print()

# ================================================================================
# CRITICAL ISSUES DETAIL
# ================================================================================

print("[5/7] Identifying CRITICAL normalization issues...")
print()

# Sort by absolute max value (worst first)
critical_features.sort(key=lambda x: max(abs(x['min']), abs(x['max'])), reverse=True)

print("TOP 20 WORST NORMALIZED FEATURES (Wall Street Red Flags)")
print("=" * 100)
print(f"{'#':<3} | {'Feature':<40} | {'Min':>12} | {'Max':>12} | {'Std':>10} | {'Category':<15} | {'Fix':<15}")
print("-" * 100)

for i, f in enumerate(critical_features[:20], 1):
    print(f"{i:<3} | {f['feature'][:40]:<40} | {f['min']:>12.2f} | {f['max']:>12.2f} | {f['std']:>10.2f} | {f['category']:<15} | {f['normalization']:<15}")

print("-" * 100)
print()

# ================================================================================
# WALL STREET VERDICT
# ================================================================================

print("[6/7] WALL STREET VERDICT")
print()

total_features = len(feature_analysis)
needs_norm_count = sum(1 for f in feature_analysis if f['needs_norm'])
critical_count = sum(1 for f in feature_analysis if f['needs_norm'] and f['severity'] in ['CRITICAL', 'HIGH'])
ok_count = total_features - needs_norm_count

print("=" * 80)
print("NORMALIZATION HEALTH CHECK")
print("=" * 80)
print()
print(f"  Total Features:          {total_features}")
print(f"  Already OK:              {ok_count} ({ok_count/total_features*100:.1f}%)")
print(f"  Needs Normalization:     {needs_norm_count} ({needs_norm_count/total_features*100:.1f}%)")
print(f"  CRITICAL Issues:         {critical_count} ({critical_count/total_features*100:.1f}%)")
print()

# Severity breakdown
print("Severity Breakdown:")
for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'OK', 'UNKNOWN']:
    count = sum(1 for f in feature_analysis if f['severity'] == severity and f['needs_norm'])
    if count > 0:
        print(f"  [{severity:8}] {count:3} features need fixing")

print()

# Final verdict
if critical_count > 10:
    print("[CRITICAL] NORMALIZATION TRÈS PROBLÉMATIQUE")
    print("           Le réseau de neurones va avoir du mal à apprendre!")
    print("           SOLUTION: Appliquer normalisation avant training")
    verdict = "CRITICAL"
elif critical_count > 5:
    print("[WARNING] NORMALIZATION À CORRIGER")
    print("          Plusieurs features mal normalisées")
    verdict = "WARNING"
elif needs_norm_count > 10:
    print("[ATTENTION] NORMALIZATION À AMÉLIORER")
    print("            Quelques features hors plage")
    verdict = "ATTENTION"
else:
    print("[OK] NORMALIZATION ACCEPTABLE")
    print("     La plupart des features sont bien normalisées")
    verdict = "OK"

print()

# ================================================================================
# SOLUTION: NORMALIZATION CODE
# ================================================================================

print("[7/7] Generating WALL STREET GRADE normalization solution...")
print()

# Group features by normalization type needed
norm_groups = defaultdict(list)
for f in feature_analysis:
    if f['needs_norm']:
        norm_groups[f['normalization']].append(f['feature'])

print("=" * 80)
print("RECOMMENDED NORMALIZATIONS BY TYPE")
print("=" * 80)
print()

for norm_type, features in norm_groups.items():
    print(f"[{norm_type.upper()}] - {len(features)} features")
    for feat in features[:5]:  # Show first 5
        print(f"    - {feat}")
    if len(features) > 5:
        print(f"    - ... and {len(features) - 5} more")
    print()

# ================================================================================
# GENERATE FIX CODE
# ================================================================================

if args.fix:
    print("=" * 80)
    print("NORMALIZATION FIX CODE (Copy to feature_engineering.py)")
    print("=" * 80)
    print()

    print("""
def normalize_features_wallstreet(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Wall Street Grade Feature Normalization
    Transforms ALL features to appropriate ranges for neural networks
    '''
    df_norm = df.copy()

    # 1. PRICE RAW FEATURES -> Min-Max to [0, 1]
    price_cols = [c for c in df.columns if any(p in c.lower() for p in ['close', 'open', 'high', 'low', 'sma_', 'ema_'])]
    for col in price_cols:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # 2. PERCENTAGE FEATURES -> Divide by 100
    pct_cols = [c for c in df.columns if any(p in c.lower() for p in ['rsi_', 'stochastic_', 'mfi_', 'williams_r'])]
    for col in pct_cols:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col] / 100.0

    # 3. VOLUME FEATURES -> Log + Z-score
    vol_cols = [c for c in df.columns if 'volume' in c.lower() and 'ratio' not in c.lower()]
    for col in vol_cols:
        if col in df_norm.columns:
            df_norm[col] = np.log1p(df_norm[col])
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
                df_norm[col] = df_norm[col].clip(-3, 3)  # Clip outliers

    # 4. ATR/VOLATILITY -> Percentile rank [0, 1]
    atr_cols = [c for c in df.columns if 'atr_' in c.lower() or 'vol_' in c.lower()]
    for col in atr_cols:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].rank(pct=True)

    # 5. OSCILLATORS -> Z-score clipped
    osc_cols = [c for c in df.columns if any(p in c.lower() for p in ['macd', 'momentum', 'roc', 'cci', 'adx'])]
    for col in osc_cols:
        if col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
                df_norm[col] = df_norm[col].clip(-3, 3)

    # 6. FINAL CLIP: Everything in [-10, 10] as safety net
    for col in df_norm.columns:
        df_norm[col] = df_norm[col].clip(-10, 10)

    return df_norm
""")

# ================================================================================
# SAVE REPORT
# ================================================================================

if args.output:
    report_df = pd.DataFrame(feature_analysis)
    report_df.to_csv(args.output, index=False)
    print(f"\n[SAVED] Report saved to: {args.output}")

# Save summary
summary_path = analysis_dir / "normalization_audit_summary.json"
summary = {
    'date': datetime.now().isoformat(),
    'total_features': total_features,
    'ok_count': ok_count,
    'needs_norm_count': needs_norm_count,
    'critical_count': critical_count,
    'verdict': verdict,
    'critical_features': [f['feature'] for f in critical_features[:20]],
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[SAVED] Summary saved to: {summary_path}")

print()
print("=" * 100)
print("NORMALIZATION AUDIT COMPLETE")
print("=" * 100)
print()

# Final recommendation
print("RECOMMANDATION FINALE:")
print("-" * 50)
if verdict == "CRITICAL":
    print("1. URGENT: Ajouter normalize_features_wallstreet() dans feature_engineering.py")
    print("2. Appeler cette fonction APRÈS calculate_all_features()")
    print("3. Re-run les diagnostics pour vérifier")
    print("4. PUIS lancer le training")
elif verdict == "WARNING":
    print("1. Ajouter normalisation pour les features CRITICAL")
    print("2. Monitorer les observations pendant le training")
else:
    print("1. Les features sont relativement bien normalisées")
    print("2. Optionnel: ajouter normalisation pour améliorer convergence")

print()
print("Pour générer le code de fix:")
print("  python analysis/normalization_audit.py --fix")
