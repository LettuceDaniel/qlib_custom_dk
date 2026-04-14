import os
import pandas as pd
from pathlib import Path
from qlib.data import D

# Import from existing module (shared, no duplication)
from qlib_research.train_valid_backtest.data.cache import get_h5_data

try:
    import tomllib as toml
except ImportError:
    import toml


class HDF5DataLoader:
    def __init__(self, col_config, h5_path):
        self.h5_path = Path(h5_path)
        self.col_config = col_config
        self._data = None

    def load(self, instruments=None, start_time=None, end_time=None):
        if self._data is None:
            self._data = get_h5_data(str(self.h5_path))
            self._data = self._data.sort_index()

        df = self._data.copy()

        if instruments is not None:
            if isinstance(instruments, str):
                instruments = [instruments]
            mask = df.index.get_level_values("instrument").isin(instruments)
            df = df.loc[mask].sort_index()

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

        feature_cols = self.col_config.get("feature", [])
        label_cols = self.col_config.get("label", [])

        missing_feat = set(feature_cols) - set(df.columns)
        missing_label = set(label_cols) - set(df.columns)
        if missing_feat or missing_label:
            raise KeyError(
                f"Columns not found in H5: feature={missing_feat}, label={missing_label}"
            )

        feat_idx = pd.MultiIndex.from_tuples(
            [("feature", c) for c in feature_cols], names=["field", "item"]
        )
        label_idx = pd.MultiIndex.from_tuples(
            [("label", c) for c in label_cols], names=["field", "item"]
        )
        all_idx = pd.MultiIndex.from_tuples(
            feat_idx.tolist() + label_idx.tolist(), names=["field", "item"]
        )

        result = df[feature_cols + label_cols].copy().set_axis(all_idx, axis=1)
        return result


def load_backtest_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config",
        "env.toml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    config = toml.loads(content)
    return config


def load_backtest_config_with_overrides(config_override=None):
    """load_backtest_config() + YAML config override 통합.

    Args:
        config_override: dict, YAML config의 data 섹션에서 추출.
            keys: backtest_start, backtest_end, h5_path, label_col

    Returns:
        dict: merged backtest config (4회 중복 제거를 위해 pipeline.py에서 1회만 호출)
    """
    config = load_backtest_config()
    if config_override:
        data_cfg = config_override.get("data", {})
        if "backtest_start" in data_cfg:
            config["period"]["start"] = data_cfg["backtest_start"]
        if "backtest_end" in data_cfg:
            config["period"]["end"] = data_cfg["backtest_end"]
        if "h5_path" in data_cfg:
            config["data"]["h5_path"] = data_cfg["h5_path"]
        if "label_col" in data_cfg:
            config["data"]["label_col"] = data_cfg["label_col"]
    return config


BENCHMARK_DIR = Path("/workspace/qlib_research/qlib_data/benchmark")
BENCHMARK_CSV = BENCHMARK_DIR / "spy_benchmark.csv"


def load_spy_benchmark_from_csv(test_dates):
    """Load SPY benchmark returns from pre-downloaded CSV.

    Args:
        test_dates: list of pd.Timestamp - dates to extract benchmark returns for

    Returns:
        pd.Series with date index and daily return values, aligned to test_dates
    """
    if not BENCHMARK_CSV.exists():
        return None

    df = pd.read_csv(BENCHMARK_CSV, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # Create a Series of returns indexed by date
    returns = df["return"]

    # Reindex to test_dates (fill missing dates with 0)
    returns = returns.reindex(test_dates).fillna(0)
    return returns


def load_backtest_data(backtest_config):
    h5_path = backtest_config["data"]["h5_path"]
    df = get_h5_data(h5_path)
    labels_df = df.reset_index()[["datetime", "instrument", backtest_config["data"].get("label_col", "LABEL")]].copy()
    labels_df = labels_df.rename(columns={backtest_config["data"].get("label_col", "LABEL"): "LABEL"})

    dt_index = df.index.get_level_values("datetime")
    all_dates = sorted(dt_index.unique())
    test_dates = [
        d
        for d in all_dates
        if pd.Timestamp(backtest_config["period"]["start"])
        <= d
        <= pd.Timestamp(backtest_config["period"]["end"])
    ]
    if not test_dates:
        raise ValueError(
            f"No test dates in period {backtest_config['period']['start']} ~ "
            f"{backtest_config['period']['end']}"
        )

    start_dt = pd.Timestamp(backtest_config["period"]["start"])
    end_dt = pd.Timestamp(backtest_config["period"]["end"])

    prices = df.loc[
        (slice(start_dt, end_dt), slice(None)),
        "RET",
    ].unstack(level="instrument")

    benchmark_returns = None
    if backtest_config["benchmark"].get("instrument"):
        # First try: load from pre-downloaded CSV (SPY only for now)
        if backtest_config["benchmark"].get("instrument") == "SPY":
            benchmark_returns = load_spy_benchmark_from_csv(test_dates)
            if benchmark_returns is not None:
                print(f"  Benchmark: loaded SPY from CSV ({len(benchmark_returns)} days)")

        # Fallback: use qlib D.features
        if benchmark_returns is None:
            try:
                benchmark_inst = backtest_config["benchmark"]["instrument"]
                benchmark_data = D.features(
                    [benchmark_inst],
                    ["$close"],
                    pd.Timestamp(backtest_config["period"]["start"])
                    - pd.Timedelta(days=30),
                    pd.Timestamp(backtest_config["period"]["end"]),
                    freq="day",
                    disk_cache=True,
                )
                benchmark_df = benchmark_data["$close"].unstack(level="instrument")[benchmark_inst]
                benchmark_df.index = pd.DatetimeIndex(benchmark_df.index).floor("D")
                benchmark_returns = benchmark_df.pct_change().reindex(test_dates).fillna(0)
                print(f"  Benchmark: loaded {benchmark_inst} from qlib ({len(benchmark_returns)} days)")
            except Exception as e:
                print(f"  Warning: Could not load benchmark: {e}")

    return df, labels_df, test_dates, prices, benchmark_returns