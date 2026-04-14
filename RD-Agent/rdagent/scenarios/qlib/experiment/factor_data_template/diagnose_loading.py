#!/usr/bin/env python3
"""
Diagnose data loading performance
"""

import time
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DATA LOADING PERFORMANCE DIAGNOSTIC")
print("=" * 60)

# Test 1: H5 file loading
print("\n[Test 1] Loading H5 file...")
h5_path = Path("/workspace/qlib_data/h5_data/daily_pv_all.h5")

start = time.time()
df = pd.read_hdf(h5_path)
h5_time = time.time() - start

print(f"  Time: {h5_time:.4f}s")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Test 2: Qlib data handler initialization
print("\n[Test 2] Initializing Qlib data handler...")
import sys

sys.path.insert(0, "/workspace/RD-Agent/rdagent/scenarios/qlib/experiment/model_template")

import qlib
from qlib.data.dataset.handler import DataHandlerLP

start = time.time()
qlib.init(provider_uri="/workspace/qlib_data/us_data", region="us")
qlib_init_time = time.time() - start
print(f"  Qlib init time: {qlib_init_time:.4f}s")

# Test 3: DataHandlerLP initialization (lightweight)
handler_config = {
    "start_time": "2014-01-01",
    "end_time": "2014-01-31",  # Small range for fast test
    "instruments": ["AAPL", "MSFT", "GOOG"],  # 3 instruments
    "data_loader": {
        "class": "rdagent.scenarios.qlib.experiment.model_template.hdf5_data_loader.HDF5DataLoader",
        "kwargs": {
            "col_config": {"feature": ["OPEN", "HIGH", "LOW", "RET", "VOL"], "label": ["LABEL"]},
            "h5_path": "/workspace/qlib_data/h5_data/daily_pv_all.h5",
        },
    },
    "infer_processors": [
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "learn_processors": [{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
}

start = time.time()
handler = DataHandlerLP(**handler_config)
handler_init_time = time.time() - start
print(f"  Handler init time: {handler_init_time:.4f}s")

# Test 4: Dataset preparation (full pipeline)
print("\n[Test 3] Full dataset preparation...")
from qlib.data.dataset import TSDatasetH

dataset_config = {
    "handler": {"class": "DataHandlerLP", "module_path": "qlib.contrib.data.handler", "kwargs": handler_config},
    "segments": {
        "train": ["2014-01-01", "2014-01-31"],
        "valid": ["2014-02-01", "2014-02-28"],
        "test": ["2014-03-01", "2014-03-31"],
    },
    "step_len": 20,
}

start = time.time()
dataset = TSDatasetH(**dataset_config)
dataset_prep_time = time.time() - start
print(f"  Dataset prep time: {dataset_prep_time:.4f}s")

# Test 5: Full simulation of workspace.execute()
print("\n[Test 4] Simulating workspace.execute()...")

# Create temporary workspace-like environment
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv

import shutil
import uuid

temp_workspace = Path(f"/tmp/test_workspace_{uuid.uuid4().hex}")
temp_workspace.mkdir(parents=True, exist_ok=True)

# Copy H5 to workspace
start = time.time()
shutil.copy(h5_path, temp_workspace / "daily_pv.h5")
copy_time = time.time() - start
print(f"  Copy H5: {copy_time:.4f}s")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"H5 Loading:            {h5_time:.4f}s")
print(f"Qlib Init:            {qlib_init_time:.4f}s")
print(f"Handler Init:         {handler_init_time:.4f}s")
print(f"Dataset Prep:        {dataset_prep_time:.4f}s")
print(f"H5 Copy:             {copy_time:.4f}s")
print(f"Total (no Qlib init): {h5_time + handler_init_time + dataset_prep_time + copy_time:.4f}s")

# Clean up
import tempfile

shutil.rmtree(temp_workspace)
print(f"\nCleaned up temp workspace: {temp_workspace}")
print("\n✓ All diagnostics completed!")
