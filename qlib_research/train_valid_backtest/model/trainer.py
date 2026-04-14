import os
import json
import math
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

from torch.utils.data import DataLoader
from qlib_research.train_valid_backtest.data.dataset import (
    TimeSeriesDataset,
    create_date_grouped_dataloader,
    collate_fn,
)
from qlib_research.train_valid_backtest.evaluation.ic import compute_daily_rank_ic
from qlib_research.train_valid_backtest.model.losses import (
    HuberPearsonLoss,
    PairwiseRankingLoss,
    listwise_ranking_loss,
)


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def train(self, seed=42):
        config = self.config
        # Bug 6: Ensure internal seed matches the one used for training
        self.seed = seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.create_model()
        self.model = self.model.to(self.device)

        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        valid_dataset = TimeSeriesDataset(
            self.X_valid, self.y_valid, self.valid_end_timestamps
        )

        batch_size = config["training"]["batch_size"]
        loss_name = config["training"].get("loss", "mse")
        use_date_grouped = config["training"].get(
            "use_date_grouped", True
        )  # Default: use DateGroupedBatchSampler

        g = torch.Generator()
        g.manual_seed(seed)

        if use_date_grouped:
            print(f"  Using DateGroupedBatchSampler (loss={loss_name}, seed={seed})")
            # Bug 8: Pass seed for reproducibility
            train_loader = create_date_grouped_dataloader(
                train_dataset,
                self.train_dates,
                batch_size,
                shuffle=True,
                num_workers=0,
                seed=seed,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=g,
            )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,  # Bug #3: Apply same collate_fn to valid_loader for consistency
        )

        # Loss and Criterion setup
        pearson_weight = config["training"].get("pearson_weight", 0.1)
        if loss_name == "huber":
            criterion = nn.HuberLoss()
        elif loss_name == "huber_pearson":
            criterion = HuberPearsonLoss(huber_delta=1.0, pearson_weight=pearson_weight)
        elif loss_name == "pairwise_ranking":
            loss_kwargs = config["training"].get("loss_kwargs", {})
            margin = loss_kwargs.get("margin", 1.0)
            use_sigmoid = loss_kwargs.get("use_sigmoid", True)
            criterion = PairwiseRankingLoss(margin=margin, use_sigmoid=use_sigmoid)
        elif loss_name == "listwise_ranking":
            criterion = None
        else:
            criterion = nn.MSELoss()

        # Bug 1: Flexible validation loss criterion
        val_loss_name = config["training"].get("val_loss", loss_name)
        if val_loss_name == "huber":
            val_criterion = nn.HuberLoss()
        elif val_loss_name == "huber_pearson":
            val_criterion = HuberPearsonLoss(
                huber_delta=1.0, pearson_weight=pearson_weight
            )
        elif val_loss_name == "mse":
            val_criterion = nn.MSELoss()
        else:
            val_criterion = nn.MSELoss()

        optimizer_name = config["training"].get("optimizer", "adam")
        weight_decay = config["training"].get("weight_decay", 1e-5)
        gradient_clip = config["training"].get("gradient_clip", 1.0)
        warmup_epochs = config["training"].get("warmup_epochs", 0)
        epochs = config["training"]["epochs"]
        early_stop = config["training"]["early_stop"]

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config["training"]["lr"],
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["training"]["lr"],
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        min_lr = config["training"].get("min_lr", 1e-6)

        # Bug 2: Fix potential division by zero
        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                denom = epochs - warmup_epochs
                if denom <= 0:
                    return 1.0
                progress = (epoch - warmup_epochs) / denom
                return max(
                    min_lr / config["training"]["lr"],
                    0.5 * (1 + math.cos(math.pi * progress)),
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Bug 2 & 4: Determine validation metric and its behavior
        val_metric_name = config["training"].get(
            "val_metric",
            "rank_ic"
            if loss_name in ("pairwise_ranking", "listwise_ranking")
            else "loss",
        )
        higher_is_better = val_metric_name in ("rank_ic",)

        best_metric = -float("inf") if higher_is_better else float("inf")
        patience_counter = 0
        nan_counter = 0
        best_model_state = None
        min_delta = float(config["training"].get("min_delta", 1e-4))

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_data in train_loader:
                batch_X, batch_y = (
                    batch_data[0].to(self.device),
                    batch_data[1].to(self.device),
                )
                optimizer.zero_grad()
                # Bug #9: view(-1) flattens all dims. Only safe for output_dim=1.
                # If multi-dimensional output (e.g. output_dim>1), use squeeze(-1) or explicit reshape.
                outputs = self.model(batch_X).view(-1)

                # Bug 7: Add warning for NaN outputs
                if torch.isnan(outputs).any():
                    print(
                        f"  Warning: NaN detected in model outputs at epoch {epoch + 1}"
                    )
                    continue

                if loss_name == "pairwise_ranking":
                    loss = criterion(outputs, batch_y)
                elif loss_name == "listwise_ranking":
                    temperature = config["training"].get("listwise_temperature", 1.0)
                    loss = listwise_ranking_loss(outputs, batch_y, temperature)
                else:
                    loss = criterion(outputs, batch_y)

                if torch.isnan(loss):
                    print(f"  Warning: NaN detected in loss at epoch {epoch + 1}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / num_batches if num_batches > 0 else float("nan")
            if num_batches == 0:
                print(f"  All batches produced NaN — stopping training at epoch {epoch + 1}")
                break

            # Unified validation loop
            self.model.eval()
            current_val_metric = None
            val_loss_log = ""

            if val_metric_name == "rank_ic":
                current_val_metric = compute_daily_rank_ic(
                    self.model, valid_loader, self.device, min_samples=10
                )
                val_loss_log = f"Val Rank IC: {current_val_metric:.4f}"
            else:
                # Calculate validation loss
                val_loss_sum = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch_data in valid_loader:
                        batch_X, batch_y = (
                            batch_data[0].to(self.device),
                            batch_data[1].to(self.device),
                        )
                        outputs = self.model(batch_X).view(-1)
                        loss = val_criterion(outputs, batch_y)
                        if not torch.isnan(loss):
                            val_loss_sum += loss.item()
                            num_val_batches += 1
                current_val_metric = (
                    val_loss_sum / num_val_batches
                    if num_val_batches > 0
                    else float("nan")
                )
                val_loss_log = f"Val: {current_val_metric:.6f}"

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | Train: {train_loss:.6f} | {val_loss_log} | LR: {current_lr:.6f}",
                flush=True,
            )

            # Unified Early Stopping check
            if np.isnan(current_val_metric):
                nan_counter += 1
                print(f"  NaN validation metric at epoch {epoch + 1} (nan_counter={nan_counter})")
                if nan_counter >= early_stop:
                    print(f"Early stopping due to {nan_counter} consecutive NaN epochs.")
                    break
            else:
                nan_counter = 0
                is_best = False
                if higher_is_better:
                    if current_val_metric > best_metric + min_delta:
                        is_best = True
                else:
                    if current_val_metric < best_metric - min_delta:
                        is_best = True

                if is_best:
                    best_metric = current_val_metric
                    patience_counter = 0
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop:
                        print(
                            f"Early stopping at epoch {epoch + 1}. Best {val_metric_name}: {best_metric:.6f}"
                        )
                        break

            scheduler.step()

        if best_model_state is None:
            raise ValueError("Training failed - no valid model saved")

        self.best_model_state = best_model_state
        self.best_val_metric = best_metric
        self.model.load_state_dict(self.best_model_state)
        print(f"\nBest validation {val_metric_name}: {self.best_val_metric:.6f}")

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        config = self.config

        model_info = {
            "model_state_dict": self.best_model_state,
            "model_config": config["model"],
            "feature_cols": config["data"]["feature_cols"],
            "label_cols": config["data"]["label_cols"],
            "zscore_mean": self.zscore_mean,
            "zscore_std": self.zscore_std,
            "train_start": config["data"]["train_start"],
            "train_end": config["data"]["train_end"],
            "val_start": config["data"]["valid_start"],
            "val_end": config["data"]["valid_end"],
            "best_val_metric": float(self.best_val_metric),
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "seed": self.seed if hasattr(self, "seed") else None,
        }

        model_path = os.path.join(
            output_dir,
            f"model_seed{self.seed}.pt" if hasattr(self, "seed") else "model.pt",
        )
        torch.save(model_info, model_path)
        print(f"Model saved to: {model_path}")

        def _to_list(val):
            if isinstance(val, dict):
                return {k: _to_list(v) for k, v in val.items()}
            if isinstance(val, (np.ndarray, np.number)):
                return val.tolist()
            if isinstance(val, torch.Tensor):
                return val.cpu().numpy().tolist()
            return val

        scaler_info = {
            "zscore_mean": _to_list(self.zscore_mean),
            "zscore_std": _to_list(self.zscore_std),
            "feature_cols": config["data"]["feature_cols"],
            "label_cols": config["data"]["label_cols"],
            "num_timesteps": config["model"].get("num_timesteps"),
            "clip_range": config["data"].get("clip_range", [-3, 3]),
        }
        if hasattr(self, "market_cols") and self.market_cols:
            scaler_info["market_cols"] = self.market_cols
            scaler_info["market_mean"] = _to_list(self.market_mean)
            scaler_info["market_std"] = _to_list(self.market_std)
            scaler_info["market_norm_method"] = "median_mad"  # robust z-score with MAD * 1.4826
        if hasattr(self, "raw_cols") and self.raw_cols:
            scaler_info["raw_cols"] = self.raw_cols

        scaler_path = os.path.join(output_dir, "model_scaler.json")
        with open(scaler_path, "w") as f:
            json.dump(scaler_info, f, indent=2)
        print(f"Scaler saved to: {scaler_path}")

        config_path = os.path.join(output_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config["model"], f, indent=2)
        print(f"Config saved to: {config_path}")
