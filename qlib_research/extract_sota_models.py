"""
SOTA Model Extractor
Extract SOTA models from mlflow.db and match to RD-Agent_workspace folders.

Workflow:
1. Read potential_sota_model_result.csv (SOTA filter criteria applied)
2. Query mlflow.db for IC/ICIR metrics of each SOTA recorder_id
3. Scan all RD-Agent_workspace folders for matching IC/ICIR (dual verification)
4. Copy model files (model.py, conf_baseline_factors_model.yaml, hdf5_data_loader.py) to final_models/models/

Usage:
    python extract_sota_models.py
"""

import os
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path


# Configuration
MLFLOW_DB_PATH = "/workspace/git_ignore_folder/log_mlruns/mlflow.db"
SOTA_CSV_PATH = "/workspace/git_ignore_folder/log_all_model_results/potential_sota_model_result.csv"
RD_AGENT_WORKSPACE_PATH = "/workspace/git_ignore_folder/RD-Agent_workspace"
FINAL_MODELS_PATH = "/workspace/final_models/models"

# SOTA Filter Thresholds
IC_THRESHOLD = 0.01
RANK_IC_THRESHOLD = 0.015
MAX_DRAWDOWN_THRESHOLD = -0.25

# Files to copy from each matched folder
FILES_TO_COPY = [
    "conf_baseline_factors_model.yaml",
    "model.py",
    "hdf5_data_loader.py",
]


def get_sota_recorder_ids():
    """Read SOTA recorder_ids from potential_sota_model_result.csv"""
    recorder_ids = set()
    with open(SOTA_CSV_PATH, 'r') as f:
        for line in f.readlines()[1:]:  # skip header
            parts = line.strip().split(',')
            if len(parts) >= 7:
                recorder_ids.add(parts[2])
    return recorder_ids


def get_run_metrics_from_mlflow(recorder_ids):
    """Query mlflow.db for IC and ICIR of each recorder_id"""
    conn = sqlite3.connect(MLFLOW_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT r.run_uuid, m.key, m.value
        FROM runs r
        JOIN metrics m ON r.run_uuid = m.run_uuid
        WHERE r.lifecycle_stage = 'active'
        AND m.key IN ('IC', 'ICIR')
    """)

    run_metrics = {}
    for run_uuid, key, value in cursor.fetchall():
        if run_uuid not in run_metrics:
            run_metrics[run_uuid] = {}
        run_metrics[run_uuid][key] = float(value)

    conn.close()
    return run_metrics


def match_folders_to_sota(recorder_ids, run_metrics):
    """Match RD-Agent_workspace folders to SOTA recorder_ids using IC+ICIR dual verification"""
    folder_to_sota = {}

    for folder in sorted(os.listdir(RD_AGENT_WORKSPACE_PATH)):
        qlib_res = os.path.join(RD_AGENT_WORKSPACE_PATH, folder, "qlib_res.csv")
        if not os.path.exists(qlib_res):
            continue

        ic_val, icir_val = None, None
        with open(qlib_res) as f:
            for line in f:
                line = line.strip()
                if line.startswith('IC,'):
                    parts = line.split(',')
                    if len(parts) >= 2 and parts[1]:
                        ic_val = float(parts[1])
                elif line.startswith('ICIR,'):
                    parts = line.split(',')
                    if len(parts) >= 2 and parts[1]:
                        icir_val = float(parts[1])

        if ic_val is None or icir_val is None:
            continue

        matched = []
        for rid in recorder_ids:
            if rid not in run_metrics:
                continue
            ic_match = abs(ic_val - run_metrics[rid]['IC']) < 0.0001
            icir_match = abs(icir_val - run_metrics[rid]['ICIR']) < 0.0001
            if ic_match and icir_match:
                matched.append(rid)

        if matched:
            folder_to_sota[folder] = matched

    return folder_to_sota


def copy_model_files():
    """Main execution: extract and copy SOTA models to final_models/models/"""
    print("=== SOTA Model Extractor ===\n")

    # Step 1: Get SOTA recorder_ids
    sota_recorder_ids = get_sota_recorder_ids()
    print(f"Step 1: Found {len(sota_recorder_ids)} SOTA models in {SOTA_CSV_PATH}")

    # Step 2: Get metrics from mlflow.db
    run_metrics = get_run_metrics_from_mlflow(sota_recorder_ids)
    print(f"Step 2: Retrieved metrics for {len(run_metrics)} runs from mlflow.db")

    # Step 3: Match folders to SOTA recorder_ids
    folder_to_sota = match_folders_to_sota(sota_recorder_ids, run_metrics)
    print(f"Step 3: Matched {len(folder_to_sota)} folders to SOTA recorder_ids")

    # Step 4: Copy files
    print(f"\nStep 4: Copying model files to {FINAL_MODELS_PATH}\n")

    os.makedirs(FINAL_MODELS_PATH, exist_ok=True)

    total_copied = 0
    for folder, matched_rids in folder_to_sota.items():
        src_folder = os.path.join(RD_AGENT_WORKSPACE_PATH, folder)
        dst_folder = os.path.join(FINAL_MODELS_PATH, folder)
        os.makedirs(dst_folder, exist_ok=True)

        for fname in FILES_TO_COPY:
            src_file = os.path.join(src_folder, fname)
            dst_file = os.path.join(dst_folder, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                total_copied += 1
            else:
                print(f"  WARNING: {fname} not found in {folder}")

        print(f"  {folder}/ ({len(matched_rids)} SOTA matches)")

    print(f"\n=== Complete ===")
    print(f"Total folders created: {len(folder_to_sota)}")
    print(f"Total files copied: {total_copied}")


if __name__ == "__main__":
    copy_model_files()