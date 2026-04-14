#!/usr/bin/env python3
"""
Test HDF5DataLoader directly
"""

import sys

sys.path.insert(0, "/workspace/RD-Agent/rdagent/scenarios/qlib/experiment/model_template")

from hdf5_data_loader import HDF5DataLoader
from pathlib import Path

# Create HDF5DataLoader instance
col_config = {"feature": ["OPEN", "HIGH", "LOW", "RET", "VOL"], "label": ["LABEL"]}

h5_path = "/workspace/qlib_data/h5_data/daily_pv_all.h5"

print(f"Testing HDF5DataLoader with:")
print(f"  h5_path: {h5_path}")
print(f"  col_config: {col_config}")

try:
    loader = HDF5DataLoader(col_config=col_config, h5_path=h5_path)

    # Test loading all data
    print("\n--- Test 1: Load all data ---")
    df_all = loader.load()
    print(f"Shape: {df_all.shape}")
    print(f"Columns: {df_all.columns.tolist()}")
    print(f"Index names: {df_all.columns.names}")

    # Test loading with time range
    print("\n--- Test 2: Load with time range ---")
    df_time = loader.load(start_time="2014-01-01", end_time="2014-12-31")
    print(f"Shape: {df_time.shape}")

    # Test loading with instruments
    print("\n--- Test 3: Load with instruments ---")
    df_inst = loader.load(instruments=["AAPL", "MSFT"])
    print(f"Shape: {df_inst.shape}")

    # Test loading with both
    print("\n--- Test 4: Load with time and instruments ---")
    df_both = loader.load(start_time="2014-01-01", end_time="2014-12-31", instruments=["AAPL", "MSFT"])
    print(f"Shape: {df_both.shape}")

    print("\nSUCCESS: All HDF5DataLoader tests passed!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
