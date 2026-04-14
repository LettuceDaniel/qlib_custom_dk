import inspect
import pandas as pd
import numpy as np

from qlib_research.train_valid_backtest.model.trainer import BaseTrainer
from qlib_research.train_valid_backtest.data.dataloader import HDF5DataLoader
from qlib_research.train_valid_backtest.data.preprocessing import (
    dropna_label,
    robust_zscore_norm,
    csrank_norm,
    create_sequences_from_df,
    load_market_features,
    normalize_market_features,
)
from qlib_research.train_valid_backtest.data.cache import load_model_class


class ModelTrainer(BaseTrainer):
    def __init__(self, config, model_folder, seed=42):
        super().__init__(config)
        self.model_folder = model_folder
        self.seed = seed
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.zscore_mean = None
        self.zscore_std = None
        self.market_mean = None
        self.market_std = None
        self.market_cols = None
        self.best_model_state = None
        self.best_val_metric = None

    def prepare_data(self):
        config = self.config
        data_config = config["data"]
        model_config = config["model"]

        h5_path = data_config["h5_path"]
        col_config = {
            "feature": data_config["feature_cols"],
            "label": data_config["label_cols"],
        }

        loader = HDF5DataLoader(col_config, h5_path)
        df = loader.load(
            start_time=data_config["train_start"], end_time=data_config["valid_end"]
        )

        print(f"Raw data shape: {df.shape}")

        df = dropna_label(df)
        print(f"After dropping NaN labels: {df.shape}")

        df, self.zscore_mean, self.zscore_std = robust_zscore_norm(
            df,
            "feature",
            data_config["train_start"],
            data_config["train_end"],
            clip_outlier=True,
        )
        print(f"Feature normalization params (fit on train period):")
        print(f"  Feature mean: {self.zscore_mean}")
        print(f"  Feature std:  {self.zscore_std}")

        # Bug 1 & 4: Use pd.Timestamp for IndexSlice and apply csrank_norm to full df
        # Use train_end as split boundary (data already sliced to valid_end during load)
        train_end = pd.Timestamp(data_config["train_end"])

        df = csrank_norm(df, "label")

        market_cols = config.get("market_features", {}).get("columns", [])
        if market_cols:
            raw_cols = config.get("market_features", {}).get("raw_columns", [])
            print(f"Loading market features: {market_cols} (raw: {raw_cols})")
            market_df = load_market_features(
                market_cols,
                start_time=data_config["train_start"],
                end_time=data_config["valid_end"],
                raw_cols=raw_cols,
            )
            market_df, self.market_mean, self.market_std = normalize_market_features(
                market_df,
                data_config["train_start"],
                data_config["train_end"],
                raw_cols=raw_cols,
            )
            self.market_cols = market_cols
            self.raw_cols = raw_cols
            print(f"Market normalization params (RobustZScore: median/MAD, fit on train period):")
            print(f"  Market median: {self.market_mean}")
            print(f"  Market MAD:   {self.market_std}")

            market_feat_cols = [("feature", c) for c in market_cols]
            market_dates = market_df.index
            date_level = df.index.get_level_values("datetime")
            # Normalize both to datetime64[ns] to ensure consistent comparison
            date_level_norm = (
                pd.to_datetime(date_level).normalize().astype("datetime64[ns]")
            )
            market_dates_norm = (
                pd.to_datetime(market_dates).normalize().astype("datetime64[ns]")
            )

            # Build lookup dict from normalized market dates to values
            # Include raw_cols in the loop to merge them too
            all_market_cols = list(market_cols) + list(raw_cols) if raw_cols else list(market_cols)
            for col in all_market_cols:
                date_to_val = dict(zip(market_dates_norm, market_df[col].values))
                mapped = pd.Series(date_level_norm).map(date_to_val)
                new_col = ("feature", col)
                df[new_col] = mapped.values
            market_feat_cols = [("feature", c) for c in all_market_cols]
            df = df.dropna(subset=market_feat_cols)
            print(f"After market features merge: {df.shape}")

        X_all, y_all, end_timestamps, instruments_arr = create_sequences_from_df(
            df, num_timesteps=model_config["num_timesteps"]
        )
        print(f"Total sequences: {len(X_all)}")

        # Store actual feature count after all market features are merged
        self.actual_num_features = X_all.shape[-1]
        print(f"Actual num_features: {self.actual_num_features} (config: {model_config.get('num_features', 'N/A')})")

        if len(X_all) == 0:
            raise ValueError("No valid sequences created!")

        end_timestamps_arr = np.array(end_timestamps, dtype="datetime64[ns]")
        # Bug 6: Sort once
        sorted_indices = np.argsort(end_timestamps_arr)
        X_all = X_all[sorted_indices]
        y_all = y_all[sorted_indices]
        end_timestamps_arr = end_timestamps_arr[sorted_indices]
        instruments_arr = instruments_arr[sorted_indices]

        # Bug 3: Use numpy datetime for vectorized comparison
        train_end_np = train_end.to_datetime64()

        train_mask = end_timestamps_arr <= train_end_np
        valid_mask = end_timestamps_arr > train_end_np

        self.X_train = X_all[train_mask]
        self.y_train = y_all[train_mask]
        self.X_valid = X_all[valid_mask]
        self.y_valid = y_all[valid_mask]

        self.train_dates = end_timestamps_arr[train_mask]
        self.valid_end_timestamps = end_timestamps_arr[valid_mask]

        # Bug #4: Warn if NaT timestamps found (they are silently dropped in training)
        nat_mask = pd.isnull(self.valid_end_timestamps)
        if nat_mask.any():
            print(f"  Warning: {nat_mask.sum()} NaT timestamps in valid set — will be skipped in training")

        # Bug 5: Store both for completeness
        self.train_instruments = instruments_arr[train_mask]
        self.instrument_list = instruments_arr[valid_mask]

        print(f"Train: {len(self.X_train)}, Valid: {len(self.X_valid)}")

    def create_model(self):
        config = self.config
        model_config = config["model"]

        model_class = load_model_class(self.model_folder)

        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        model_kwargs = {
            k: v
            for k, v in model_config.items()
            if k in valid_params and k != "model_class"
        }

        # Use actual_num_features if set (reflects any market features added dynamically)
        if hasattr(self, "actual_num_features"):
            model_kwargs["num_features"] = self.actual_num_features

        self.model = model_class(**model_kwargs)

        print(f"Model: {model_class.__name__}, kwargs: {model_kwargs}")
