import os
import json
import glob
import torch
import numpy as np
import pandas as pd

from qlib_research.train_valid_backtest.data.preprocessing import (
    load_market_features,
    normalize_market_features,
)
from qlib_research.train_valid_backtest.data.cache import create_model_instance


def get_inference_dates(df, test_dates, num_timesteps):
    """Calculate inference date range (with warmup period) from test dates.

    Args:
        df: DataFrame with datetime index
        test_dates: list of test dates
        num_timesteps: number of timesteps for model input

    Returns:
        list: dates to run inference on (includes warmup period)
    """
    calendar = sorted(df.index.get_level_values(0).unique())
    start_idx = max(0, calendar.index(test_dates[0]) - num_timesteps + 1)
    end_idx = calendar.index(test_dates[-1]) + 1
    return calendar[start_idx:end_idx]


def load_model_and_scaler(model_dir, seed=None):
    if seed is not None:
        model_path = os.path.join(model_dir, f"model_seed{seed}.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.pt")
    else:
        # Try model_seed*.pt if exists (ensemble/pre-trained mode)
        seed_files = sorted(glob.glob(os.path.join(model_dir, "model_seed*.pt")))
        if seed_files:
            model_path = seed_files[0]
        else:
            model_path = os.path.join(model_dir, "model.pt")

    scaler_path = os.path.join(model_dir, "model_scaler.json")
    model_config_path = os.path.join(model_dir, "model_config.json")

    with open(scaler_path) as f:
        scaler = json.load(f)

    with open(model_config_path) as f:
        model_config = json.load(f)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_info = checkpoint.get("model_state_dict", checkpoint)

    return model_info, scaler, model_config


def run_inference(model, device, df, dates, scaler, features, num_timesteps):
    model.eval()
    predictions = {}

    zscore_mean = np.array(scaler["zscore_mean"]).reshape(1, 1, -1)
    zscore_std = np.array(scaler["zscore_std"]).reshape(1, 1, -1)
    clip_range = scaler["clip_range"]

    # Bug #1: Assert shape match between scaler and features
    assert len(scaler["zscore_mean"]) == len(features), \
        f"Scaler features ({len(scaler['zscore_mean'])}) != inference features ({len(features)})"

    market_cols = scaler.get("market_cols", [])
    raw_cols = scaler.get("raw_cols", [])
    market_mean = scaler.get("market_mean")
    market_std = scaler.get("market_std")
    market_aligned = None
    if market_cols and market_mean and market_std:
        market_raw = load_market_features(market_cols, raw_cols=raw_cols)
        # normalize_market_features computes: (x - median) / MAD
        # Inference uses identical formula for consistent normalization
        for col in market_cols:
            market_raw[col] = (market_raw[col] - market_mean[col]) / market_std[col]
        # raw_cols are not normalized (kept as-is with median=0, MAD=1)
        market_aligned = market_raw

    df_reset = df.reset_index()
    df_pivot = df_reset.pivot(index="datetime", columns="instrument", values=features)
    df_pivot = df_pivot.sort_index(axis=1)
    pivot_index = pd.to_datetime(df_pivot.index)
    dates_ts = [pd.Timestamp(d).normalize() for d in dates]
    dt_to_idx = {pd.Timestamp(dt).normalize(): i for i, dt in enumerate(pivot_index)}

    arrays = [df_pivot[feat].values for feat in features]
    arr_all = np.stack(arrays, axis=2)

    for day_idx, date in enumerate(dates_ts):
        if day_idx % 50 == 0:
            print(f"  [Inference] {day_idx + 1}/{len(dates)} days (warmup+test)")

        end_idx = dt_to_idx.get(date)
        if end_idx is None:
            continue
        start_idx = max(0, end_idx - num_timesteps + 1)

        arr_window = arr_all[start_idx : end_idx + 1]

        if arr_window.shape[0] < num_timesteps:
            continue

        arr_window = arr_window.transpose(1, 0, 2)
        valid_mask = ~(np.isnan(arr_window).any(axis=(1, 2)))
        arr_window = arr_window[valid_mask]

        if arr_window.shape[0] == 0:
            continue

        all_instruments = df_pivot[features[0]].columns.tolist()
        valid_instruments = [all_instruments[i] for i, v in enumerate(valid_mask) if v]

        arr_window = arr_window.astype(np.float32)
        arr_window = (arr_window - zscore_mean) / zscore_std
        arr_window = np.clip(arr_window, clip_range[0], clip_range[1])
        arr_window = np.nan_to_num(arr_window, nan=0.0, posinf=0.0, neginf=0.0)

        if market_aligned is not None:
            window_dates = pivot_index[start_idx : end_idx + 1]
            # Bug #7: Normalize both indices for consistent date matching
            window_dates_norm = pd.to_datetime(window_dates).normalize()
            market_window = market_aligned.loc[market_aligned.index.isin(window_dates_norm)].sort_index()
            # Bug #5: Raise error if market features are unavailable (silent skip hides shape mismatch bugs)
            assert len(market_window) == num_timesteps, (
                f"Market feature length mismatch: got {len(market_window)}, expected {num_timesteps} "
                f"(market_cols={market_cols}, date={date})"
            )
            market_vals = market_window[market_cols].values.astype(np.float32)
            market_tiled = np.tile(market_vals, (arr_window.shape[0], 1, 1))
            # For raw_cols, no normalization (use 0 median, 1 MAD so value stays as-is)
            if raw_cols:
                raw_market = market_window[raw_cols].values.astype(np.float32)
                raw_tiled = np.tile(raw_market, (arr_window.shape[0], 1, 1))
                arr_window = np.concatenate([arr_window, market_tiled, raw_tiled], axis=2)
            else:
                arr_window = np.concatenate([arr_window, market_tiled], axis=2)

        tensor = torch.from_numpy(arr_window).float().to(device)
        with torch.no_grad():
            preds = model(tensor).cpu().numpy().flatten()

        predictions[date] = pd.DataFrame(
            {"instrument": valid_instruments, "score": preds}
        )

    return predictions


def compute_ensemble_predictions(
    model_files,
    df,
    scaler,
    model_class,
    model_config,
    device,
    test_dates,
):
    """다중 시드 모델의 예측을 rank averaging으로 앙상블."""
    features = scaler["feature_cols"]
    num_timesteps = scaler["num_timesteps"]
    inference_dates = get_inference_dates(df, test_dates, num_timesteps)

    all_predictions = {}

    for model_file in model_files:
        print(f"  Running inference with {os.path.basename(model_file)}...")
        seed_name = (
            os.path.basename(model_file).replace("model_seed", "").replace(".pt", "")
        )

        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        model_info = checkpoint.get("model_state_dict", checkpoint)

        model = create_model_instance(model_info, model_class, model_config)
        model = model.to(device)

        preds = run_inference(
            model, device, df, inference_dates, scaler, features, num_timesteps
        )
        preds = {k: v for k, v in preds.items() if k in test_dates}

        for date, pred_df in preds.items():
            if date not in all_predictions:
                all_predictions[date] = []
            pred_df_copy = pred_df.copy()
            pred_df_copy["seed"] = seed_name
            all_predictions[date].append(pred_df_copy)

    print("  Computing ensemble ranks...")
    ensemble_predictions = {}

    for date, model_preds in all_predictions.items():
        if not model_preds:
            continue

        rank_dfs = []
        for seed_pred in model_preds:
            seed_pred = seed_pred.copy()
            seed_pred["rank"] = seed_pred["score"].rank(ascending=False)
            rank_dfs.append(seed_pred[["instrument", "rank"]])

        avg_rank = pd.concat(rank_dfs).groupby("instrument")["rank"].mean()

        ensemble_predictions[date] = pd.DataFrame(
            {
                "instrument": avg_rank.index,
                "score": -avg_rank.values,
            }
        )

    print(f"  Completed ensemble for {len(ensemble_predictions)} days")
    return ensemble_predictions
