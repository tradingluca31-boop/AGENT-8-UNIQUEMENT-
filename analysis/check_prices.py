"""
QUICK CHECK - Verify price data is correct
"""

import sys
from pathlib import Path

# Add paths
analysis_dir = Path(__file__).resolve().parent
agent8_root = analysis_dir.parent
env_dir = agent8_root / "environment"
goldrl_root = Path("C:/Users/lbye3/Desktop/GoldRL")
goldrl_v2 = goldrl_root / "AGENT" / "AGENT 8" / "ALGO AGENT 8 RL" / "V2"

sys.path.insert(0, str(goldrl_root))
sys.path.insert(0, str(goldrl_v2))
sys.path.insert(0, str(env_dir))

print("="*60)
print("PRICE DATA CHECK")
print("="*60)

from data_loader import load_data_agent8

df_full, auxiliary_data = load_data_agent8()

print(f"\ndf_full shape: {df_full.shape}")
print(f"\nFirst 20 columns:")
for i, col in enumerate(df_full.columns[:20]):
    print(f"  {i}: {col}")

# Find OHLCV columns
print("\n" + "="*60)
print("OHLCV COLUMNS SEARCH")
print("="*60)

ohlcv_cols = [col for col in df_full.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
print(f"\nFound {len(ohlcv_cols)} OHLCV columns:")
for col in ohlcv_cols[:10]:
    print(f"  {col}: {df_full[col].iloc[1000]:.4f}")

# The correct columns should contain actual Gold prices (~800-2000)
print("\n" + "="*60)
print("LOOKING FOR GOLD PRICES (should be 800-2000)")
print("="*60)

for col in df_full.columns:
    if 'close' in col.lower() or 'open' in col.lower():
        sample_val = df_full[col].iloc[1000]
        if 500 < sample_val < 3000:
            print(f"  [FOUND!] {col}: {sample_val:.2f}")

# Check raw XAUUSD data
print("\n" + "="*60)
print("RAW XAUUSD DATA CHECK")
print("="*60)

if 'xauusd_raw' in auxiliary_data:
    for tf, data in auxiliary_data['xauusd_raw'].items():
        if data is not None and len(data) > 0:
            print(f"\n{tf} data:")
            print(f"  Shape: {data.shape}")
            print(f"  Columns: {list(data.columns[:5])}")
            if 'close' in data.columns:
                print(f"  Close sample: {data['close'].iloc[1000]:.2f}")

print("\n" + "="*60)
print("CHECK COMPLETE")
print("="*60)
