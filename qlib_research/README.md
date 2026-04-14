# qlib_research

qlib 기반 모델 학습 및 백테스트 독립 프로젝트.

---

## 프로젝트 구조

```
qlib_research/
├── qlib/                    # qlib 라이브러리 (bundled)
├── qlib_data/               # 데이터
│   ├── us_data/             # qlib 백테스트용 주식 데이터
│   └── h5_data/             # 학습용 피처 데이터
├── train_valid_backtest/    # 학습 + 백테스트 파이프라인
│   ├── pipeline.py           # 메인 엔트리 포인트 (3가지 모드)
│   ├── workflow/             # 학습 워크플로우
│   │   └── training.py       # train_single_model, train_all_seeds_and_filter
│   ├── model/                # 모델 추론
│   │   └── inference.py      # run_inference, compute_ensemble_predictions
│   ├── backtest/             # 백테스트 엔진
│   │   └── runner.py        # run_backtest_pair
│   ├── data/                 # 데이터 로딩/전처리
│   │   ├── cache.py          # H5 캐시, 모델 로드/인스턴스화
│   │   └── dataloader.py     # 백테스트 설정, 데이터 로딩
│   └── evaluation/           # 평가 지표
│       └── reporting.py      # build_results_and_report
├── models/                  # 기존 모델 (#1 ~ #17)
├── models_tuned/            # 신규/개선 모델 (TCNAlpha_v2, TCNAlpha_v3, BiGRU 등)
├── config/                  # 환경 설정
├── requirements.txt         # Python 의존성
└── model_analysis.md        # 모델 분석 결과 기록
```

---

## 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Conda 환경 사용 (권장)

```bash
conda activate rdagent4qlib
```

환경 경로: `/venv/rdagent4qlib`, Python 3.10

### 3. 패키지 설치 (개발 모드)

```bash
# 프로젝트 루트에서 실행
cd /workspace/qlib_research
pip install -e .
```

이 명령어로 `qlib_research` 패키지를 개발 모드로 설치하면:
- `python -m train_valid_backtest.pipeline` 형태로 모듈 실행 가능
- 코드 수정 시 재설치 불필요

---

## 사용법

### 1. 기본 학습 (Single Model)

```bash
# 프로젝트 루트에서 실행
cd /workspace/qlib_research
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate rdagent4qlib

# 기본 명령
python -m train_valid_backtest.pipeline --config <model_folder>/model_train.yaml --seed <number>

# 예시: TCNAlpha_v3 시드 1 학습
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --seed 1

# 예시: TCNAlpha_v2 시드 1 학습
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v2/model_train.yaml --seed 1
```

### 2. 기존 모델로 백테스트만 실행

```bash
# --no-train: 학습 생략, 기존 model_seed{seed}.pt 사용
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --seed 1 --no-train
```

### 3. 앙상블 백테스트 (Pre-trained)

```bash
# model_seed*.pt 파일들을 로드하여 앙상블 실행 (기본 출력: ensemble_results.csv)
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --ensemble

# 출력 파일명 변경
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --ensemble --ensemble-results my_ensemble.csv
```

### 4. Train-Ensemble 모드 (15 seeds → IC ≥ 0.015 필터링 → 앙상블)

```bash
# 15개 시드 학습 후 validation IC ≥ 0.015 인 모델만 선택하여 앙상블
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --train-ensemble

# 시드 수/IC threshold 커스터마이즈 (model_train.yaml의 training section에서도 설정 가능)
python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --train-ensemble --num-seeds 20
```

### 5. 다중 시드 병렬 학습

```bash
# 3개 시드 병렬 학습 (백그라운드)
for seed in 1 2 3; do
  source /workspace/miniconda/etc/profile.d/conda.sh
  conda activate rdagent4qlib
  python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --seed $seed --log-dir models_tuned/TCNAlpha_v3/log_train &
done
wait  # 모든 시드 완료 대기
```

### 6. 결과 비교 (V2 vs V3)

```bash
# V2와 V3 결과 비교
python -c "
import pandas as pd
v2 = pd.read_csv('models_tuned/TCNAlpha_v2/results_combined.csv')
v3 = pd.read_csv('models_tuned/TCNAlpha_v3/results_combined.csv')
print('=== V2 Seed 1-3 ===')
print(v2[v2['seed'].isin([1,2,3])][['seed','IC','ICIR','Rank IC','Rank ICIR','information_ratio','max_drawdown']])
print('=== V3 Seed 1-3 ===')
print(v3[v3['seed'].isin([1,2,3])][['seed','IC','ICIR','Rank IC','Rank ICIR','information_ratio','max_drawdown']])
"

---

## 로그 및 출력 관리

### 로그 파일 자동 저장

학습 시 로그가 자동으로 `log_train/` 폴더에 저장됩니다:
- 파일명 형식: `s{seed}_es{early_stop}_{YYYYMMDD_HHMMSS}.log`
- 예: `s1_es10_20260411_110605.log`
- 출력 리다이렉션不要 — 코드 내에서 자동 처리

### 병렬 학습 (여러 시드 동시 실행)

```bash
# 3개 시드 병렬 학습 (백그라운드)
for seed in 1 2 3; do
  source /workspace/miniconda/etc/profile.d/conda.sh
  conda activate rdagent4qlib
  python -m train_valid_backtest.pipeline --config models_tuned/TCNAlpha_v3/model_train.yaml --seed $seed --log-dir models_tuned/TCNAlpha_v3/log_train &
done
wait  # 모든 시드 완료 대기
```

- 로그 파일: `<model_folder>/log_train/s{seed}_es{early_stop}_{timestamp}.log`
- 학습 결과: `results_combined.csv` (자동 누적)

---

## 학습 결과 확인

### 결과 파일 (모두 model 폴더 내에 저장)

| 파일 | 설명 |
|------|------|
| `results_combined.csv` | 모든 시드 결과 누적 (seed, timestamp 포함) |
| `train_ensemble_results.csv` | --train-ensemble 모드 결과 |
| `seed_ic_results.csv` | --train-ensemble 모드 시드별 IC 결과 |
| `log_train/` | 학습 로그 파일들 (s{seed}_es{es}_{timestamp}.log) |
| `model_seed{seed}.pt` | 학습된 모델 |
| `model_scaler.json` | 피처 정규화 파라미터 |
| `model_config.json` | 모델 설정 |

### 주요 지표

| 지표 | 설명 |
|------|------|
| IC | Information Coefficient (예측 정확도) |
| ICIR | IC / IC 표준편차 (안정성) |
| Rank IC | 순위 기반 IC (강건함) |
| Rank ICIR | Rank IC 비율 |
| Excess Return | 초과 수익률 (%) |
| Information Ratio (IR) | risk-adjusted 수익률 |
| Max Drawdown | 최대 손실폭 (%) |

---

## model_train.yaml 설정 가이드

```yaml
model:
  model_class: TCNAlpha_v3     # Python 클래스명
  num_features: 5             # 입력 피처 수
  num_timesteps: 20           # 시퀀스 길이
  hidden_dim: 128             # 은닉층 차원
  num_blocks: 4               # TCN 블록 수
  kernel_size: 3              # 커널 크기
  dropout: 0.3                # 드롭아웃율
  dilation_base: 2             # dilation 베이스

data:
  h5_path: /workspace/qlib_research/qlib_data/h5_data/daily_pv_all.h5   # 학습 데이터 경로
  feature_cols: [OPEN, HIGH, LOW, RET, VOL]
  label_cols: [LABEL]
  train_start: "2014-01-01"
  train_end: "2022-12-31"
  valid_start: "2023-01-01"
  valid_end: "2024-12-31"

# Market features (선택, 모델에 따라 사용)
# market_features:
#   columns:
#     - VIX                    # VIX 지수 (정규화됨)
#     - TNX                    # 미국 재무부 수익률 (정규화됨)
#     - TNX_CHG_3M            # 3개월 수익률 변화 (raw_columns로 유지)
#   raw_columns:
#     - TNX_CHG_3M

training:
  batch_size: 512
  epochs: 100
  early_stop: 10              # 조기 종료 patience

  # Learning Rate Schedule (Cosine Annealing + Warmup)
  warmup_epochs: 3            # warmup 에포크 수
  lr: 0.002                   # 초기 learning rate
  min_lr: 0.000001            # 최소 learning rate
  val_metric: rank_ic         # 검증 지표

  # Optimizer 설정
  optimizer: adamw           # adam | adamw
  weight_decay: 0.0001       # 가중치 감쇠
  gradient_clip: 2.0         # 그래디언트 클리핑

  # Loss 설정 (pairwise_ranking / listwise_ranking / huber / mse)
  loss: pairwise_ranking
  loss_kwargs:
    margin: 0.05
    use_sigmoid: false
```

### 주요 설정 설명

| 설정 | 설명 | 권장값 |
|------|------|--------|
| warmup_epochs | Cosine annealing 전 linear warmup 에포크 | 3 |
| lr | 초기 학습률 | 0.002 (V3), 0.001 (V2) |
| optimizer | 최적화 알고리즘 | adamw 권장 |
| gradient_clip | 그래디언트 클리핑으로 학습 안정화 | 2.0 |
| val_metric | 검증 기준 지표 | rank_ic (pairwise/listwise) |
| loss | 손실 함수 | pairwise_ranking (순위 학습) |

### early_stop 설정 참고

| 값 | 설명 |
|----|------|
| 7 | 너무 짧음, premature 종료 (IC 저하 원인) |
| 8 | 적정 (TCNAlpha_v2 기본값) |
| 10 | 여유 있음 (권장) |

---

## 모델 폴더 구조

### 모델 폴더 구조

```
<model_folder>/
├── model_architecture.py
├── model_train.yaml
├── model_seed{seed}.pt          # 학습된 모델
├── model_scaler.json            # 피처 정규화 파라미터
├── model_config.json            # 모델 설정
├── results_combined.csv         # 모든 시드 결과 누적 (single/train-ensemble)
├── train_ensemble_results.csv   # train-ensemble 모드 결과
├── seed_ic_results.csv          # train-ensemble 시드별 IC 결과
└── log_train/                   # 학습 로그 (s{seed}_es{es}_{timestamp}.log)
```

### 예시: TCNAlpha_v3

```
models_tuned/TCNAlpha_v3/
├── model_architecture.py
├── model_train.yaml
├── model_seed*.pt               # 시드별 학습 모델
├── model_scaler.json            # 피처 정규화 파라미터
├── model_config.json            # 모델 설정
├── results_combined.csv         # 모든 시드 결과 누적
├── train_ensemble_results.csv   # train-ensemble 결과
├── seed_ic_results.csv          # 시드별 IC 결과
└── log_train/                   # 학습 로그
```

---

## 문제 해결

### torch module not found

```bash
conda activate rdagent4qlib
```

### 재학습 시 결과 파일 삭제

```bash
# 모델만 삭제 (결과는 results_combined.csv에 누적됨)
rm -f model_seed*.pt

# 결과까지 모두 삭제
rm -f model_seed*.pt results_combined.csv train_ensemble_results.csv seed_ic_results.csv ensemble_results.csv
```

### 실행 중인 프로세스 확인/종료

```bash
ps aux | grep pipeline
kill <PID>
```