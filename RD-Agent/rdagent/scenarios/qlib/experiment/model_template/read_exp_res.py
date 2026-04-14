import os
import pickle
import shutil
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import qlib
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI to shared mlruns folder
mlruns_dir = os.environ.get("RDAGENT_MLRUNS_DIR", "/workspace/log_mlruns")
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{mlruns_dir}/mlflow.db")
client = MlflowClient(tracking_uri=tracking_uri)

qlib.init()

# ── 1) 최신 recorder 찾기 (성공한 run만) ──
experiments = client.search_experiments(max_results=100)

latest_run = None
latest_end_time = None
latest_experiment_name = None

for exp in experiments:
    runs = client.search_runs([exp.experiment_id], "", max_results=100)
    for run in runs:
        # Skip failed runs
        if run.info.status != "FINISHED":
            continue
        end_time = run.info.end_time
        if end_time is None:
            continue
        if latest_end_time is None or end_time > latest_end_time:
            latest_end_time = end_time
            latest_run = run
            latest_experiment_name = exp.name

if latest_run is None:
    print("No FINISHED recorders found")
    exit(1)

print(f"Latest run: {latest_run.info.run_id}")
print(f"Status: {latest_run.info.status}")

# ── 2) metrics 가져오기 ──
metrics = latest_run.data.metrics
print(f"Metrics: {metrics}")

# qlib_res.csv 저장
metrics_series = pd.Series(metrics)
output_path = Path(__file__).resolve().parent / "qlib_res.csv"
metrics_series.to_csv(output_path)
print(f"Output has been saved to {output_path}")

# ret.pkl 저장 (artifacts에서)
try:
    local_dir = Path(__file__).resolve().parent
    client.download_artifacts(latest_run.info.run_id, "portfolio_analysis/report_normal_1day.pkl", str(local_dir))
    ret_df = pd.read_pickle(local_dir / "report_normal_1day.pkl")
    ret_df.to_pickle(local_dir / "ret.pkl")
    print(f"Artifact saved to ret.pkl")
except Exception as e:
    print(f"Could not load portfolio_analysis artifact: {e}")

# ── 3) 전체 이력 누적: all_model_results.csv ──
history_path = (
    Path(os.environ.get("RDAGENT_MODEL_DATA_DIR", "/workspace/qlib_data/model_data")) / "all_model_results.csv"
)
history_path.parent.mkdir(parents=True, exist_ok=True)

new_row = {
    "timestamp": datetime.fromtimestamp(latest_run.info.start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "experiment": latest_experiment_name,
    "recorder_id": latest_run.info.run_id,
    "run_id": os.environ.get("RDAGENT_RUN_ID", ""),
    "IC": metrics.get("IC", None),
    "ICIR": metrics.get("ICIR", None),
    "Rank IC": metrics.get("Rank IC", None),
    "Rank ICIR": metrics.get("Rank ICIR", None),
    "excess_return_without_cost": metrics.get("1day.excess_return_without_cost.annualized_return", None),
    "excess_return_with_cost": metrics.get("1day.excess_return_with_cost.annualized_return", None),
    "information_ratio": metrics.get("1day.excess_return_without_cost.information_ratio", None),
    "max_drawdown": metrics.get("1day.excess_return_with_cost.max_drawdown", None),
}

new_df = pd.DataFrame([new_row])

if history_path.exists():
    existing = pd.read_csv(history_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
else:
    combined = new_df

combined = combined.round(5)
# Strip whitespace from experiment column to ensure consistent formatting
if "experiment" in combined.columns:
    combined["experiment"] = combined["experiment"].str.strip()
combined.to_csv(history_path, index=False)
print(f"Appended to {history_path} (total {len(combined)} records)")

# ── 4) Cleanup: Remove large artifacts (dataset, config) to save space ──
try:
    artifact_uri = latest_run.info.artifact_uri
    # Convert file:// URI to local path
    if artifact_uri.startswith("file://"):
        artifact_dir = Path(artifact_uri.removeprefix("file://"))
    else:
        artifact_dir = Path(artifact_uri)

    # Remove dataset (typically 80+ MB)
    dataset_path = artifact_dir / "dataset"
    if dataset_path.exists():
        if dataset_path.is_file():
            size = dataset_path.stat().st_size / 1024 / 1024
            dataset_path.unlink()
            print(f"Deleted dataset ({size:.1f} MB)")
        elif dataset_path.is_dir():
            size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()) / 1024 / 1024
            shutil.rmtree(dataset_path)
            print(f"Deleted dataset/ ({size:.1f} MB)")

    # Remove config (typically 100+ MB)
    config_path = artifact_dir / "config"
    if config_path.exists():
        if config_path.is_file():
            size = config_path.stat().st_size / 1024 / 1024
            config_path.unlink()
            print(f"Deleted config ({size:.1f} MB)")
        elif config_path.is_dir():
            size = sum(f.stat().st_size for f in config_path.rglob("*") if f.is_file()) / 1024 / 1024
            shutil.rmtree(config_path)
            print(f"Deleted config/ ({size:.1f} MB)")

    print(f"Cleanup completed for run {latest_run.info.run_id}")
except Exception as e:
    print(f"Cleanup warning: {e}")
