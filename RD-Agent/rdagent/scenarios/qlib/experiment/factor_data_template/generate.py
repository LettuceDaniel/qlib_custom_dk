import qlib
import pandas as pd
from pathlib import Path

qlib.init(provider_uri="/workspace/qlib_data/us_data", region="us")

from qlib.data import D


def apply_split_adjustment(data):
    """
    Backward Adjustment:
    최신 가격을 기준으로 과거 가격을 수정합니다.

    가정:
    - index level 0: datetime
    - index level 1: symbol
    - $factor: 분할 발생일에만 분할 비율 기록, 그 외는 1
      예) 2-for-1 split 이면 factor=2
    """
    data = data.copy()

    # 0이나 NaN 방지
    data["$factor"] = data["$factor"].replace(0, 1).fillna(1)

    # 종목별로 날짜순 정렬이 되어 있다는 가정
    # 당일 factor는 당일이 아니라 "더 과거 데이터"에만 적용되어야 하므로 shift 필요
    adj_factor = data.groupby(level=1)["$factor"].transform(
        lambda x: x.iloc[::-1].cumprod().shift(1, fill_value=1).iloc[::-1]
    )

    # 가격 조정
    for col in ["$open", "$close", "$high", "$low"]:
        data[col] = data[col] / adj_factor

    # 거래량 조정
    data["$volume"] = data["$volume"] * adj_factor

    return data


instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
print("Loading full data...")
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2014-01-01":].sort_index()
print(f"Original data shape: {data.shape}")

print("Applying split adjustment...")
data = apply_split_adjustment(data)
print(f"Adjusted data shape: {data.shape}")

# ── 1. OCHLV ratios + label 계산 ──
# Price ratio features
data["OPEN"] = data["$open"] / data["$close"]
data["HIGH"] = data["$high"] / data["$close"]
data["LOW"] = data["$low"] / data["$close"]

# RET (1-day return, split-adjusted) — groupby approach for proper alignment
close_series = data["$close"]
ret_grouped = close_series.groupby(level=1).transform(lambda x: x / x.shift(1) - 1)
data["RET"] = ret_grouped

# VOL (relative to 20-day mean, split-adjusted)
vol_series = data["$volume"]
vol_ma20_grouped = vol_series.groupby(level=1).transform(lambda x: x.rolling(20, min_periods=1).mean())
data["VOL"] = vol_series / vol_ma20_grouped

# LABEL (5-day forward return, split-adjusted) — shift(-5) per instrument
label_grouped = close_series.groupby(level=1).transform(lambda x: x.shift(-5) / x - 1)
data["LABEL"] = label_grouped

# ── 2. Features + Label 컬럼만 선택 후 저장 ──
cols_to_save = ["OPEN", "HIGH", "LOW", "RET", "VOL", "LABEL"]
output_dir = Path("/workspace/qlib_data/h5_data")
output_dir.mkdir(parents=True, exist_ok=True)
data[cols_to_save].to_hdf(str(output_dir / "daily_pv_all.h5"), key="data")
print(f"Saved to {output_dir / 'daily_pv_all.h5'} with columns: {cols_to_save}")


# ── 3. Debug data: same format ──
# Use already-adjusted full data, sliced to period — this ensures splits outside
# the period (e.g. AAPL 4:1 in 2020) are already accounted for.
print("\nSlicing debug data from adjusted full data...")
debug_data = data.loc["2018-01-01":"2019-12-31"].copy()

print(f"debug_data.index.names: {debug_data.index.names}")
print(f"debug_data.shape: {debug_data.shape}")

# Get instruments that actually have data in this period
valid_instruments = debug_data.reset_index()["instrument"].unique()[:100]
print(f"valid_instruments (first 5): {valid_instruments[:5]}")

# Filter to instruments that have data in this period
debug_data_filtered = debug_data.loc[
    debug_data.index.get_level_values(1).isin(valid_instruments)
]
print(f"debug_data_filtered.shape: {debug_data_filtered.shape}")

debug_data_filtered[cols_to_save].to_hdf(str(output_dir / "daily_pv_debug.h5"), key="data")
print(f"Saved to {output_dir / 'daily_pv_debug.h5'} with columns: {cols_to_save}")
