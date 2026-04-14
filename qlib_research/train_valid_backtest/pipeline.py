#!/usr/bin/env python3
"""
Train and Backtest Pipeline for SOTA Models (Refactored)

Usage:
    python pipeline.py --config <model_folder>/model_train.yaml
    python pipeline.py --train-ensemble --config <model_folder>/model_train.yaml
    python pipeline.py --ensemble --config <model_folder>/model_train.yaml
"""

import os
import sys
import argparse
import random
import glob
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import qlib

from qlib_research.train_valid_backtest.workflow.training import (
    train_single_model,
    train_all_seeds_and_filter,
)
from qlib_research.train_valid_backtest.model.inference import (
    compute_ensemble_predictions,
    load_model_and_scaler,
    run_inference,
    get_inference_dates,
)
from qlib_research.train_valid_backtest.evaluation.ic import compute_trainer_validation_ic
from qlib_research.train_valid_backtest.evaluation.reporting import build_results_and_report
from qlib_research.train_valid_backtest.backtest.runner import run_backtest_pair
from qlib_research.train_valid_backtest.data.dataloader import (
    load_backtest_config_with_overrides,
    load_backtest_data,
)
from qlib_research.train_valid_backtest.data.cache import (
    load_model_class,
    create_model_instance,
    get_model_params,
)


def run_ensemble_pipeline(
    model_files,
    config,
    output_dir,
    results_filename,
    model_name,
    experiment,
):
    """공통 앙상블 파이프라인 (train-ensemble, ensemble 공통 사용)."""
    backtest_config = load_backtest_config_with_overrides(config)
    df, labels_df, test_dates, prices, benchmark_returns = load_backtest_data(
        backtest_config
    )

    model_info, scaler, model_config = load_model_and_scaler(output_dir)
    model_class = load_model_class(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_preds = compute_ensemble_predictions(
        model_files=model_files,
        df=df,
        scaler=scaler,
        model_class=model_class,
        model_config=model_config,
        device=device,
        test_dates=test_dates,
    )

    results_no, results_with, trades = run_backtest_pair(
        ensemble_preds, prices, test_dates, backtest_config, benchmark_returns
    )

    seed_names = [
        os.path.basename(f).replace("model_seed", "").replace(".pt", "")
        for f in model_files
    ]

    build_results_and_report(
        predictions=ensemble_preds,
        results_df_no_cost=results_no,
        results_df_with_cost=results_with,
        daily_trades_with_cost=trades,
        labels_df=labels_df,
        test_dates=test_dates,
        prices=prices,
        benchmark_returns=benchmark_returns,
        backtest_config=backtest_config,
        output_dir=output_dir,
        results_filename=results_filename,
        model_name=model_name,
        seed=f"ensemble_{len(model_files)}",
    )


def main():
    parser = argparse.ArgumentParser(description="Train and Backtest SOTA model")
    parser.add_argument("--config", required=True, help="Path to model_train.yaml")
    parser.add_argument(
        "--results",
        default="results_combined.csv",
        help="Output CSV filename (cumulative)",
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Skip training, use existing model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run ensemble backtest using model_seed*.pt files",
    )
    parser.add_argument(
        "--ensemble-results",
        default="ensemble_results.csv",
        help="Output CSV filename for ensemble results",
    )
    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        help="Train 15 seeds, filter by validation IC >= 0.015, then ensemble",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=15,
        help="Number of seeds to train in ensemble mode (default: 15)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to save training logs (default: <model_folder>/log_train)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save models and results (default: <model_folder>)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_folder = os.path.dirname(os.path.abspath(args.config))
    model_name = os.path.basename(model_folder)
    output_dir = args.output_dir if args.output_dir else model_folder

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(model_folder, "log_train")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Train and Backtest: {config['model']['model_class']}")
    print(f"{'=' * 60}\n")

    qlib.init(provider_uri="/workspace/qlib_research/qlib_data/us_data", region="us")

    # ============================================
    # Train-Ensemble Mode
    # ============================================
    if args.train_ensemble:
        num_seeds = config.get("training", {}).get("num_seeds", args.num_seeds)
        ic_threshold = config.get("training", {}).get("ic_threshold", 0.015)
        seed_range = range(1, num_seeds + 1)
        print(
            f"\n[Train-Ensemble] Training {num_seeds} seeds (IC threshold: {ic_threshold})..."
        )

        selected_seeds, results_df, confidence = train_all_seeds_and_filter(
            config,
            model_folder,
            output_dir,
            seed_range=seed_range,
            ic_threshold=ic_threshold,
        )

        if confidence == "FAIL":
            print("\nFATAL: No models passed validation IC >= 0.015 threshold.")
            seed_results_path = os.path.join(output_dir, "seed_ic_results.csv")
            results_df.to_csv(seed_results_path, index=False)
            sys.exit(1)

        print(
            f"\n[Train-Ensemble] Selected {len(selected_seeds)} models ({confidence} confidence)"
        )

        seed_results_path = os.path.join(output_dir, "seed_ic_results.csv")
        results_df.to_csv(seed_results_path, index=False)

        selected_model_files = []
        for seed, _ in selected_seeds:
            model_file = os.path.join(output_dir, f"model_seed{seed}.pt")
            if os.path.exists(model_file):
                selected_model_files.append(model_file)

        run_ensemble_pipeline(
            model_files=selected_model_files,
            config=config,
            output_dir=output_dir,
            results_filename="train_ensemble_results.csv",
            model_name=model_name,
            experiment="train_ensemble",
        )
        return

    # ============================================
    # Ensemble Mode (pre-trained)
    # ============================================
    if args.ensemble:
        print("\n[Ensemble] Running ensemble backtest with multiple seed models...")

        model_pattern = os.path.join(output_dir, "model_seed*.pt")
        model_files = sorted(glob.glob(model_pattern))
        if len(model_files) < 2:
            raise ValueError(
                f"Need at least 2 seed models for ensemble, found: {model_files}"
            )

        run_ensemble_pipeline(
            model_files=model_files,
            config=config,
            output_dir=output_dir,
            results_filename=args.ensemble_results,
            model_name=model_name,
            experiment="ensemble",
        )
        return

    # ============================================
    # Single Model Training
    # ============================================
    model_path = os.path.join(output_dir, f"model_seed{args.seed}.pt")

    early_stop = config.get("training", {}).get("early_stop", 8)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"s{args.seed}_es{early_stop}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    print(f"  Execution log: {log_path}")

    original_stdout = sys.stdout
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file

    try:
        backtest_config = load_backtest_config_with_overrides(config)
        df, labels_df, test_dates, prices, benchmark_returns = load_backtest_data(
            backtest_config
        )

        if args.no_train and os.path.exists(model_path):
            print("[1/3] Skipping training, using existing model...")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            best_val_metric = checkpoint.get(
                "best_val_metric", checkpoint.get("best_val_loss", None)
            )
            print(f"  Loaded existing model, best_val_metric: {best_val_metric}")
            val_ic_result = None
        else:
            print("[1/3] Training model...")
            trainer = train_single_model(config, model_folder, output_dir, args.seed)
            best_val_metric = trainer.best_val_metric

            val_ic_result = compute_trainer_validation_ic(trainer, labels_df)
            print(
                f"  Validation IC: {val_ic_result['IC']:.4f}, ICIR: {val_ic_result['ICIR']:.4f}, "
                f"Rank_IC: {val_ic_result['Rank_IC']:.4f}, Rank_ICIR: {val_ic_result['Rank_ICIR']:.4f}"
            )

        # ============================================
        # Load model for backtest
        # ============================================
        print("\n[2/3] Running backtest...")

        model_info, scaler, model_config = load_model_and_scaler(output_dir, args.seed)
        model_class = load_model_class(model_folder)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model_instance(model_info, model_class, model_config)
        model = model.to(device)

        print(
            f"  Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)"
        )

        features = scaler["feature_cols"]
        num_timesteps = scaler["num_timesteps"]
        inference_dates = get_inference_dates(df, test_dates, num_timesteps)

        predictions = run_inference(
            model, device, df, inference_dates, scaler, features, num_timesteps
        )

        results_no, results_with, trades = run_backtest_pair(
            predictions, prices, test_dates, backtest_config, benchmark_returns
        )

        # total_params: importlib.util 직접 로드 대신 get_model_params() 사용
        total_params = get_model_params(model_folder, model_config)

        print("\n[3/3] Calculating metrics...")

        build_results_and_report(
            predictions=predictions,
            results_df_no_cost=results_no,
            results_df_with_cost=results_with,
            daily_trades_with_cost=trades,
            labels_df=labels_df,
            test_dates=test_dates,
            prices=prices,
            benchmark_returns=benchmark_returns,
            backtest_config=backtest_config,
            output_dir=output_dir,
            results_filename=args.results,
            model_name=model_name,
            seed=args.seed,
        )
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()