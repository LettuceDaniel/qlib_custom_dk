from __future__ import annotations

import pandas as pd
from pathlib import Path


def extract_sota_run_ids(csv_path: str | Path) -> list[str]:
    df = pd.read_csv(csv_path)
    if "run_id" not in df.columns:
        return []
    return df["run_id"].dropna().tolist()
