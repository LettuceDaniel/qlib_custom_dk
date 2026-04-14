import os
import pandas as pd

# Import from existing modules (shared utilities, no duplication)
from qlib_research.train_valid_backtest.model.model_trainer import ModelTrainer
from qlib_research.train_valid_backtest.data.dataloader import load_backtest_config
from qlib_research.train_valid_backtest.data.cache import get_h5_data
from qlib_research.train_valid_backtest.evaluation.ic import compute_trainer_validation_ic


def train_single_model(config, model_folder, output_dir, seed):
    """단일 모델 학습만 수행. validation IC는 호출 측에서 계산.

    Returns:
        ModelTrainer: 학습 완료된 trainer (best_val_metric 포함)
    """
    trainer = ModelTrainer(config, model_folder, seed=seed)
    trainer.prepare_data()
    trainer.train(seed)
    trainer.save_model(output_dir)
    return trainer


def train_all_seeds_and_filter(
    config, model_folder, output_dir, seed_range=range(1, 16), ic_threshold=0.015
):
    """다중 시드 학습 후 validation IC 기반으로 필터링.

    Args:
        config: model_train.yaml 파싱 결과
        model_folder: 모델 디렉토리 경로
        output_dir: 모델 저장 디렉토리
        seed_range: 학습할 시드 범위
        ic_threshold: 통과 기준 IC threshold

    Returns:
        tuple: (selected_seeds, all_results_df, confidence)
            - selected_seeds: list of (seed, ic) tuples (상위 5개, IC 내림차순)
            - all_results_df: 모든 시드 결과 DataFrame
            - confidence: "FAIL" | "LOW" | "HIGH"
    """
    backtest_config = load_backtest_config()
    h5_path = backtest_config["data"]["h5_path"]
    label_col = config["data"].get("label_cols", ["LABEL"])[0]
    df = get_h5_data(h5_path)
    labels_df = df.reset_index()[["datetime", "instrument", label_col]].copy()
    labels_df = labels_df.rename(columns={label_col: "LABEL"})

    all_dates = sorted(df.index.get_level_values(0).unique())
    train_end = pd.Timestamp(config["data"]["train_end"])
    valid_end = pd.Timestamp(config["data"]["valid_end"])
    valid_dates = [d for d in all_dates if train_end < d <= valid_end]

    print(
        f"\nValidation period: {valid_dates[0]} to {valid_dates[-1]} ({len(valid_dates)} days)"
    )

    all_results = []
    passed = []

    for seed in seed_range:
        print(f"\n{'=' * 60}")
        print(f"Training seed {seed}/{last_seed}")
        print(f"{'=' * 60}")

        trainer = train_single_model(config, model_folder, output_dir, seed)
        result = compute_trainer_validation_ic(trainer, labels_df)
        result["seed"] = seed

        print(
            f"Seed {seed} - IC: {result['IC']:.4f}, ICIR: {result['ICIR']:.4f}, "
            f"Rank_IC: {result['Rank_IC']:.4f}, Rank_ICIR: {result['Rank_ICIR']:.4f}"
        )

        all_results.append(
            {
                "seed": seed,
                "IC": result["IC"],
                "ICIR": result["ICIR"],
                "Rank_IC": result["Rank_IC"],
                "Rank_ICIR": result["Rank_ICIR"],
            }
        )

        if result["IC"] >= ic_threshold:
            passed.append((seed, result["IC"]))
            print(f"  PASSED IC >= {ic_threshold}")
        else:
            print(f"  FAILED IC >= {ic_threshold}")

    passed.sort(key=lambda x: x[1], reverse=True)
    selected = passed[:5]

    if len(selected) == 0:
        confidence = "FAIL"
    elif len(selected) <= 2:
        confidence = "LOW"
    else:
        confidence = "HIGH"

    all_results_df = pd.DataFrame(all_results)

    total_trained = len(list(seed_range))
    last_seed = list(seed_range)[-1]
    passed_count = len(passed)
    print(f"\n{'=' * 60}")
    print("SEED SELECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total trained: {total_trained}")
    print(f"Passed (IC >= {ic_threshold}): {passed_count}")

    if confidence == "FAIL":
        print(
            f"ERROR: No models passed IC >= {ic_threshold} - Architecture review recommended - abandoning ensemble"
        )
    elif confidence == "LOW":
        print(
            f"WARNING: Only {len(selected)} model(s) passed - proceeding with ensemble but confidence is LOW"
        )
    else:
        print(
            f"Proceeding with top {len(selected)} models for ensemble (HIGH confidence)"
        )

    print(f"Selected seeds (in order): {[s for s, _ in selected]}")
    print(f"Confidence: {confidence}\n{'=' * 60}")

    return selected, all_results_df, confidence