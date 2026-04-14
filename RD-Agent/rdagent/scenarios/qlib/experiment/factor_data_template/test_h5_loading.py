#!/usr/bin/env python3
"""
Test script to verify H5 data loading works correctly
"""

import pandas as pd
from pathlib import Path

# Load H5 file
h5_path = Path("/workspace/qlib_data/h5_data/daily_pv_all.h5")

print(f"Loading H5 file: {h5_path}")
df = pd.read_hdf(h5_path)

print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index names: {df.index.names}")

# Check expected columns
feature_cols = ["OPEN", "HIGH", "LOW", "RET", "VOL"]
label_cols = ["LABEL"]

missing_feat = set(feature_cols) - set(df.columns)
missing_label = set(label_cols) - set(df.columns)

if missing_feat:
    print(f"\nERROR: Missing feature columns: {missing_feat}")
else:
    print("\nSUCCESS: All feature columns found")

if missing_label:
    print(f"ERROR: Missing label columns: {missing_label}")
else:
    print("SUCCESS: All label columns found")

# Test data slicing
print("\n--- Testing data slicing ---")
test_start = "2014-01-01"
test_end = "2014-01-31"

sliced = df.loc[test_start:test_end]
print(f"Sliced shape: {sliced.shape}")

# Test instrument filtering
test_instruments = ["AAPL", "MSFT", "GOOG"]
print(f"\nFiltering by instruments: {test_instruments}")
filtered = df[df.index.get_level_values(1).isin(test_instruments)]
print(f"Filtered shape: {filtered.shape}")

print("\nAll tests completed!")
