import os
import pandas as pd
import numpy as np
from qlib.data.dataset.processor import RobustZScoreNorm, CSRankNorm

_MARKET_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "qlib_data",
    )
)


def robust_zscore_norm(
    df, fields_group, fit_start_time, fit_end_time, clip_outlier=True
):
    processor = RobustZScoreNorm(
        fields_group=fields_group,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        clip_outlier=clip_outlier,
    )
    processor.fit(df)
    df = processor(df)
    return df, processor.mean_train.tolist(), processor.std_train.tolist()


def load_market_features(market_cols, start_time=None, end_time=None, raw_cols=None):
    csv_path = os.path.join(_MARKET_DATA_DIR, "market_daily.csv")
    market = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    market.index = pd.to_datetime(market.index).normalize()

    # All columns to load: market_cols + raw_cols (dedupe to avoid duplicate column issue)
    all_cols_raw = list(market_cols) + list(raw_cols) if raw_cols else list(market_cols)
    all_cols = list(dict.fromkeys(all_cols_raw))  # preserve order, remove duplicates

    available = set(all_cols) & set(market.columns)
    missing = set(all_cols) - available
    if missing:
        raise KeyError(f"Market cols not found in CSV: {missing}")

    if start_time is not None:
        market = market.loc[pd.Timestamp(start_time) :]
    if end_time is not None:
        market = market.loc[: pd.Timestamp(end_time)]
    return market[all_cols]


def normalize_market_features(market_df, fit_start_time, fit_end_time, raw_cols=None):
    """Normalize market features using median/MAD (RobustZScoreNorm과 동일 방식).

    For raw_cols, no normalization is applied (kept as-is).
    """
    fit_start = pd.Timestamp(fit_start_time)
    fit_end = pd.Timestamp(fit_end_time)
    fit_slice = market_df.loc[fit_start:fit_end]

    raw_cols = set(raw_cols or [])
    zscore_cols = [c for c in market_df.columns if c not in raw_cols]

    median_vals = {}
    mad_vals = {}
    normalized = market_df.copy()

    if zscore_cols:
        zs = fit_slice[zscore_cols]
        median = zs.median()
        mad = (zs - median).abs().median() * 1.4826
        mad[mad == 0] = 1.0
        normalized[zscore_cols] = (market_df[zscore_cols] - median) / mad
        median_vals = median.to_dict()
        mad_vals = mad.to_dict()

    for c in raw_cols:
        median_vals[c] = 0.0
        mad_vals[c] = 1.0

    return normalized, median_vals, mad_vals


def load_and_normalize_market_features(
    market_cols,
    raw_cols,
    fit_start_time,
    fit_end_time,
    start_time=None,
    end_time=None,
):
    """
    시장 지표를 로드하고 정규화하여 반환합니다.

    Args:
        market_cols: 사용할 시장 지표 컬럼 리스트
        raw_cols: 정규화하지 않을 컬럼 리스트 (예: TNX_CHG_3M)
        fit_start_time, fit_end_time: 정규화 파라미터 fitting 기간
        start_time, end_time: 실제 로드할 데이터 기간

    Returns:
        normalized_market: 정규화된 시장 지표 DataFrame
        median_vals: fitting 기간의 median 값
        mad_vals: fitting 기간의 MAD 값
    """
    # 1. 시장 데이터 로드
    market_df = load_market_features(
        market_cols=market_cols,
        start_time=start_time,
        end_time=end_time,
    )

    # 2. 정규화 적용
    normalized_market, median_vals, mad_vals = normalize_market_features(
        market_df=market_df,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        raw_cols=raw_cols,
    )

    return normalized_market, median_vals, mad_vals


def merge_market_to_df(df, market_normalized, market_cols, raw_cols=None):
    """
    정규화된 시장 지표를 멀티인덱스 DataFrame에 병합합니다.

    Args:
        df: (datetime, instrument) 멀티인덱스 DataFrame
        market_normalized: datetime 인덱스의 시장 지표 DataFrame
        market_cols: 병합할 컬럼 리스트 (정규화 대상)
        raw_cols: 병합할 컬럼 리스트 (비정규화, raw 상태로 유지)

    Returns:
        df: 시장 지표가 추가된 DataFrame
    """
    # Normalize date_level to date-only for consistent matching with market_normalized index
    date_level = pd.to_datetime(df.index.get_level_values("datetime")).normalize()
    all_cols = list(market_cols) + list(raw_cols) if raw_cols else list(market_cols)
    for col in all_cols:
        if col in market_normalized.columns:
            # Fast vectorized map: each date gets the same market value for all instruments
            df[("feature", col)] = date_level.map(market_normalized[col]).values

    # NaN 제거 (시장 데이터가 없는 날짜)
    feature_cols = [c for c in df.columns if c[0] == "feature"]
    df = df.dropna(subset=feature_cols)

    return df


def csrank_norm(df, fields_group):
    processor = CSRankNorm(fields_group=fields_group)
    processor.fit(df)
    df = processor(df)
    return df


def dropna_label(df):
    label_cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "label"]
    df = df.dropna(subset=label_cols)
    return df


def create_sequences_from_df(data_df, num_timesteps=20):
    features = []
    labels = []
    end_timestamps = []
    instruments_arr = []

    instruments = data_df.index.get_level_values("instrument").unique()
    feature_cols = [
        c for c in data_df.columns if isinstance(c, tuple) and c[0] == "feature"
    ]
    label_cols = [
        c for c in data_df.columns if isinstance(c, tuple) and c[0] == "label"
    ]

    for instrument in instruments:
        inst_data = data_df.xs(instrument, level="instrument")

        if len(inst_data) < num_timesteps + 1:
            continue

        inst_data = inst_data.dropna(subset=feature_cols + label_cols)
        if len(inst_data) < num_timesteps + 1:
            continue

        # Verify temporal continuity to avoid non-contiguous sequences after dropna
        timestamps = inst_data.index
        if len(timestamps) > 1:
            diff = np.diff(np.asarray(timestamps, dtype=np.int64))
            # base_diff is usually 1 day in nanoseconds
            base_diff = np.median(diff)

            # Bug Fix: Allow gaps up to 7 days (weekends and holidays) for daily data
            # 86400 * 1e9 is 1 day in nanoseconds. We allow up to 7 days.
            max_allowed_diff = max(base_diff * 1.1, 7 * 86400 * 1e9)

            if np.any(diff > max_allowed_diff):
                # Only split where the gap is truly large (e.g., stock suspension)
                breakpoints = np.where(diff > max_allowed_diff)[0] + 1
                segments = np.split(np.arange(len(inst_data)), breakpoints)
                inst_data_list = [
                    inst_data.iloc[seg]
                    for seg in segments
                    if len(seg) >= num_timesteps + 1
                ]
            else:
                inst_data_list = [inst_data]
        else:
            inst_data_list = [inst_data]

        # Bug 2: Rename inner variable to avoid shadowing 'inst_data'
        for seg_data in inst_data_list:
            feat_values = seg_data[feature_cols].values
            label_values = seg_data[label_cols].values
            inst_timestamps = seg_data.index

            # Flatten label array: keep first label column only if multiple exist
            if label_values.ndim > 1 and label_values.shape[1] > 1:
                label_values = label_values[:, 0]
            elif label_values.ndim > 1:
                label_values = label_values.flatten()
            else:
                label_values = label_values.flatten()

            # Label should be at the last feature timestep (not ahead of it)
            # so that feature T-19...T and label T are aligned (both represent info at T)
            for i in range(len(seg_data) - num_timesteps):
                seq_x = feat_values[i : i + num_timesteps]
                seq_y = label_values[
                    i + num_timesteps - 1
                ]  # align with last feature day
                seq_end_ts = inst_timestamps[i + num_timesteps - 1]

                if np.any(np.isnan(seq_x)) or np.isnan(seq_y):
                    continue

                features.append(seq_x)
                labels.append(seq_y)
                end_timestamps.append(seq_end_ts)
                instruments_arr.append(instrument)

    # Bug #2: Handle empty features case to avoid downstream shape errors
    if len(features) == 0:
        return (
            np.empty((0, num_timesteps, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            np.array([], dtype=object),
        )

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        end_timestamps,
        np.array(instruments_arr),
    )
