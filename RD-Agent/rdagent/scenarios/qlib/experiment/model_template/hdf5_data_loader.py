import os
import pandas as pd
from pathlib import Path
from qlib.data.dataset.loader import DataLoader

# Disable HDF5 file locking to avoid conflicts between multiple processes
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class HDF5DataLoader(DataLoader):
    """
    DataLoader that loads features and labels from a single H5 file.

    Parameters
    ----------
    col_config : dict
        Dictionary with 'feature' (list of column names) and 'label' (list of column names).
        Example: {"feature": ["OPEN", "HIGH", "LOW", "RET", "VOL"], "label": ["LABEL"]}
    h5_path : str or Path
        Path to the H5 file containing all columns.
    """

    def __init__(self, col_config: dict, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        self.col_config = col_config
        self._data = None
        super().__init__()  # DataLoader.__init__ takes no args

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        # 1) Load H5 once and ensure sorted index
        if self._data is None:
            self._data = pd.read_hdf(self.h5_path)
            if not self._data.index.is_monotonic_increasing:
                self._data = self._data.sort_index()

        df = self._data

        # 2) Filter by instruments
        if instruments is not None:
            if isinstance(instruments, str):
                instruments = [instruments]
            mask = df.index.get_level_values(1).isin(instruments)
            df = df.loc[mask].sort_index()

        # 3) Filter by time range using IndexSlice for MultiIndex
        def _parse_ts(val):
            s = str(val).strip()
            if s in ("", "None", "null"):
                return None
            return pd.Timestamp(s)

        start_ts = _parse_ts(start_time)
        end_ts = _parse_ts(end_time)
        if start_ts is not None and end_ts is not None:
            df = df.loc[pd.IndexSlice[start_ts:end_ts, :]]
        elif start_ts is not None:
            df = df.loc[pd.IndexSlice[start_ts:, :]]
        elif end_ts is not None:
            df = df.loc[pd.IndexSlice[:end_ts, :]]

        # 4) Select only feature + label columns and create MultiIndex for Qlib compatibility
        feature_cols = self.col_config.get("feature", [])
        label_cols = self.col_config.get("label", [])

        missing_feat = set(feature_cols) - set(df.columns)
        missing_label = set(label_cols) - set(df.columns)
        if missing_feat or missing_label:
            raise KeyError(f"Columns not found in H5: feature={missing_feat}, label={missing_label}")

        # Build MultiIndex columns: level 0 = group (feature/label), level 1 = actual column name
        feat_idx = pd.MultiIndex.from_tuples([("feature", c) for c in feature_cols], names=["field", "item"])
        label_idx = pd.MultiIndex.from_tuples([("label", c) for c in label_cols], names=["field", "item"])
        all_idx = feat_idx.append(label_idx)

        result = df[feature_cols + label_cols].copy().set_axis(all_idx, axis=1)
        return result
