#!/usr/bin/env python3
"""
Minimal Qlib test with HDF5DataLoader
"""

import qlib
from qlib.data.dataset.handler import DataHandlerLP

# Initialize Qlib
qlib.init(provider_uri="/workspace/qlib_data/us_data", region="us")

# Create minimal handler config
handler_config = {
    "start_time": "2014-01-01",
    "end_time": "2014-12-31",
    "instruments": "sp500_pit",
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

try:
    # Create handler
    handler = DataHandlerLP(**handler_config)

    print(f"Handler created successfully!")
    print(f"Data shape: {handler._data.shape}")
    print(f"Data columns: {handler._data.columns.tolist()}")
    print(f"Index names: {handler._data.columns.names}")

    print("\nSUCCESS: DataHandlerLP with HDF5DataLoader works!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
