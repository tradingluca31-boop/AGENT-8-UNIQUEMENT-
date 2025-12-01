"""
QUICK DIAGNOSTIC - Why actions don't become trades?
================================================================================
Agent chooses SELL/BUY but 0 trades are opened.
This script tests the environment directly to find the bug.
================================================================================
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
analysis_dir = Path(__file__).resolve().parent
agent8_root = analysis_dir.parent
env_dir = agent8_root / "environment"
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

sys.path.insert(0, str(goldrl_root))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(env_dir))

print("="*80)
print("QUICK DIAGNOSTIC - Why 0 trades?")
print("="*80)
print()

# Import
from trading_env import GoldTradingEnvAgent8
from data_loader import load_data_agent8
from feature_engineering import calculate_all_features

# Load data
print("[1/4] Loading data...")
df_full, auxiliary_data = load_data_agent8()
df_train = df_full.loc['2008-01-01':'2020-12-31']

for tf in ['D1', 'H1', 'M15']:
    if tf in auxiliary_data['xauusd_raw']:
        mask = (auxiliary_data['xauusd_raw'][tf].index >= '2008-01-01') & \
               (auxiliary_data['xauusd_raw'][tf].index <= '2020-12-31')
        auxiliary_data['xauusd_raw'][tf] = auxiliary_data['xauusd_raw'][tf].loc[mask]

# Calculate features FIRST (to get the aligned index)
features_df = calculate_all_features(df_train, auxiliary_data)

# FIX: Use RAW XAUUSD H1 data for prices (has real OHLCV: open, high, low, close, volume)
# The df_train only contains _close columns from different assets, NOT real OHLCV!
prices_raw = auxiliary_data['xauusd_raw']['H1'].loc['2008-01-01':'2020-12-31'].copy()

# Align prices_df with features_df (same index)
common_index = features_df.index.intersection(prices_raw.index)
prices_df = prices_raw.loc[common_index]
features_df = features_df.loc[common_index]
print(f"       [PRICES] Close sample: ${prices_df['close'].iloc[1000]:.2f}")
print(f"       [OK] Data loaded: {features_df.shape}")
print()

# Create env with VERBOSE
print("[2/4] Creating environment (VERBOSE=True)...")
env = GoldTradingEnvAgent8(
    features_df=features_df,
    prices_df=prices_df,
    initial_balance=100_000.0,
    verbose=True,  # VERBOSE to see what's happening
    training_mode=True
)
print(f"       [OK] Environment created")
print(f"       Action space: {env.action_space}")
print()

# Reset
print("[3/4] Testing manual actions...")
print("-"*60)
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial position: {env.position_side}")
print(f"Initial balance: ${env.balance:,.2f}")
print()

# Test actions directly
print("Testing 10 BUY actions (action=2)...")
print("-"*60)
for i in range(10):
    action = 2  # BUY
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}: action=BUY, position={env.position_side}, trades={len(env.trades)}, reward={reward:.4f}")
    if done:
        print("Episode done!")
        break

print()
print("-"*60)
print(f"After 10 BUY actions:")
print(f"  Position: {env.position_side} (0=flat, 1=long, -1=short)")
print(f"  Total trades: {len(env.trades)}")
print(f"  Balance: ${env.balance:,.2f}")
print("-"*60)

# Check if _open_position is being called
print()
print("[4/4] Checking _open_position method...")
print("-"*60)

# Reset and try direct call
obs, info = env.reset()
print(f"Position before: {env.position_side}")

# Get current price
current_price = env.prices_df.iloc[env.current_step]['close']
print(f"Current price: ${current_price:.2f}")

# Try to open position directly
print("Calling _open_position(side=1, price=current_price)...")
try:
    env._open_position(side=1, price=current_price)
    print(f"Position after: {env.position_side}")
    print(f"Entry price: {env.entry_price}")
except Exception as e:
    print(f"ERROR: {e}")

print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
