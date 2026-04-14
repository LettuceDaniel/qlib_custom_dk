import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr

def compute_daily_rank_ic(model, valid_loader, device, min_samples=10):
    from collections import defaultdict

    model.eval()
    date_preds = defaultdict(list)
    date_labels = defaultdict(list)

    with torch.no_grad():
        for batch_X, batch_y, batch_ts in valid_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze().cpu().numpy()
            labels = batch_y.cpu().numpy()

            for pred, label, ts in zip(outputs, labels, batch_ts):
                date_key = str(np.datetime64(ts, "D"))
                date_preds[date_key].append(pred)
                date_labels[date_key].append(label)

    daily_ics = []
    for date_key in date_preds:
        preds = np.array(date_preds[date_key])
        labels_arr = np.array(date_labels[date_key])
        if len(preds) >= min_samples:
            if np.std(preds) == 0 or np.std(labels_arr) == 0:
                continue
            ic, _ = spearmanr(preds, labels_arr)
            if not np.isnan(ic):
                daily_ics.append(ic)

    return np.mean(daily_ics) if daily_ics else 0.0


def _calc_ic(predictions, labels_df, dates, min_obs=10):
    ic_list = []
    rank_ic_list = []

    for date in dates:
        date = pd.Timestamp(date).normalize()
        pred_df = predictions.get(date)
        if pred_df is None or pred_df.empty:
            continue

        date_labels = labels_df[labels_df["datetime"] == date]
        if date_labels.empty:
            continue

        merged = pred_df.merge(date_labels, on="instrument")
        if len(merged) < min_obs:
            continue

        ic = merged["score"].corr(merged["LABEL"])
        if not np.isnan(ic):
            ic_list.append({"date": date, "ic": ic})

        ric, _ = spearmanr(merged["score"], merged["LABEL"])
        if not np.isnan(ric):
            rank_ic_list.append({"date": date, "rank_ic": ric})

    ic_df = pd.DataFrame(ic_list)
    ric_df = pd.DataFrame(rank_ic_list)

    if len(ic_df) > 0:
        mean_ic = ic_df["ic"].mean()
        std_ic = ic_df["ic"].std()
        icir = mean_ic / std_ic if std_ic > 0 else 0
    else:
        mean_ic = std_ic = icir = 0

    # Bug #4: Separate length check for ric_df (can be empty even if ic_df is not)
    if len(ric_df) > 0:
        mean_ric = ric_df["rank_ic"].mean()
        std_ric = ric_df["rank_ic"].std()
        ric_ir = mean_ric / std_ric if std_ric > 0 else 0
    else:
        mean_ric = std_ric = ric_ir = 0

    # Bug #4: Include both IC and Rank IC observation counts
    return {
        "IC": mean_ic,
        "IC_std": std_ic,
        "ICIR": icir,
        "Rank_IC": mean_ric,
        "Rank_IC_std": std_ric,
        "Rank_ICIR": ric_ir,
        "n_obs": len(ic_df),
        "n_rank_obs": len(ric_df),
    }

def compute_ic_metrics(predictions, labels_df, test_dates):
    return _calc_ic(predictions, labels_df, test_dates)

def compute_trainer_validation_ic(trainer, labels_df):
    trainer.model.eval()
    predictions_list = []
    batch_size = 2048
    for start in range(0, len(trainer.X_valid), batch_size):
        batch = torch.FloatTensor(trainer.X_valid[start : start + batch_size]).to(
            trainer.device
        )
        with torch.no_grad():
            pred = trainer.model(batch).cpu().numpy().flatten()
        predictions_list.append(pred)
    predictions = np.concatenate(predictions_list)

    predictions_dict = {}
    for i, (ts, instrument) in enumerate(
        zip(trainer.valid_end_timestamps, trainer.instrument_list)
    ):
        date = pd.Timestamp(ts)
        if date not in predictions_dict:
            predictions_dict[date] = []
        predictions_dict[date].append(
            {"instrument": instrument, "score": predictions[i]}
        )

    for date in predictions_dict:
        predictions_dict[date] = pd.DataFrame(predictions_dict[date])

    valid_dates = list(predictions_dict.keys())
    return _calc_ic(predictions_dict, labels_df, valid_dates)