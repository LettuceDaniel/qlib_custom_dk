# next_update.md — pipeline.py 버그 및 코드 중복 분석

> 분석 대상: `train_valid_backtest/pipeline.py` 및 관련 모듈

---

## 🐛 버그

### 1. `sharpe_ratio` 항상 0.0 (risk.py L73)

`risk_analysis()`에서 `sharpe_ratio: 0.0`으로 하드코딩:

```python
return {
    ...
    "sharpe_ratio": 0.0,  # Requires risk-free rate input; currently not computed
}
```

pipeline.py L395에서 `risk_excess_with_cost["sharpe_ratio"]`를 읽어오지만 항상 0.
반면 `information_ratio`는 정상 계산되므로, 실제로는 IR을 Sharpe로 표시하는 혼란 발생.

파이프라인의 최종 결과 보고서에 Sharpe ratio가 포함되어 있고, 이 값을 기반으로 모델 간 성능 비교를 하게 됨.
항상 0.0이 출력되면 사용자가 이를 "Sharpe가 실제로 0이다"라고 오해할 수 있고,
결과 CSV에 기록되면 이후 분석에서도 오류가 전파됨.
IR과 Sharpe를 동일하게 계산하는 현재 `risk_analysis()` 코드와도 불일치.

**해결**: risk-free rate을 파라미터로 받거나, 최소한 IR 값을 Sharpe에도 복사.

---

### 2. `total_params` 구할 때 캐시 무시하고 모델 모듈 재로드 (pipeline.py L447-451)

L306에서 `load_model_class(model_folder)`로 이미 로드했는데, `importlib.util`로 다시 직접 로드:

```python
model_path_py = os.path.join(model_folder, "model_architecture.py")
spec = importlib.util.spec_from_file_location("model_module", model_path_py)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
```

`data/cache.py`의 캐시를 우회함. `load_model_class()`를 사용해야 함.

---

### 3. `load_model_and_scaler(seed=None)` 엉뚱한 파일 로드 위험 (inference.py L18-19)

ensemble/train-ensemble 모드에서는 seed를 넘기지 않아 `model.pt`를 찾음.
실제로는 `model_seed*.pt` 파일만 존재.
만약 output_dir에 `model.pt` 파일이 우연히 존재하면(예: 이전 single-model 실행의 잔여 파일),
ensemble 경로에서 엉뚱한 모델을 로드하게 됨.
실제로 `torch.load`로 다시 덮어쓰기 때문에 동작은 하지만,
scaler 정보(`zscore_mean`, `zscore_std`)는 잘못된 파일에서 읽어올 수 있음.

**해결**: ensemble 경로에서는 `load_model_and_scaler()` 호출을 생략하거나, 함수가 seed 리스트를 지원하도록 수정.

---

### 4. `val_ic_result` truthiness 항상 True (pipeline.py L424-427)

`no-train` 모드에서 `val_ic_result`를 dict로 설정:

```python
val_ic_result = {"IC": None, "ICIR": None, "Rank_IC": None, "Rank_ICIR": None}
```

이후 results_row 구성 시:

```python
"Val_IC": val_ic_result["IC"] if val_ic_result else None,
```

`val_ic_result`는 항상 truthy한 dict이므로 `if val_ic_result`는 항상 `True`.
결과적으로 `None` 값들이 `results_row`에 삽입됨.
의도대로 동작하긴 하지만 truthiness 체크가 의미 없고, 빈 dict `{}`가 반환되는 경우와 구분되지 않음.

**해결**: truthiness 체크 대신 명시적으로 `val_ic_result.get("IC")` 사용하거나, `no-train`에서도 IC 계산 수행.

---

### 5. `initial_capital` 미사용 변수 (pipeline.py L373)

```python
initial_capital = backtest_config["backtest"]["initial_capital"]  # 정의 후 참조 없음
```

single model 경로에서 정의되지만 한 번도 사용되지 않음. 제거 필요.

---

### 6. `load_backtest_config()` 취약한 경로 계산 (dataloader.py L71-75)

```python
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "env.toml",
)
```

`os.path.dirname` 3번 중첩으로 프로젝트 루트를 추정. 패키지 구조 변경 시 즉시 깨짐.

**해결**: 프로젝트 루트를 상수로 정의하거나, `pathlib` 기반으로 명시적으로 탐색.

---

## 🔄 코드 중복

### 1. Backtest config 오버라이드 + `load_backtest_data` 호출 — 3회 반복 (가장 심각)

동일한 패턴이 4번 반복 (config override 4회 + load_backtest_data 3회):

| 위치 | config override 라인 | load_backtest_data 라인 |
|------|------|------|
| Train-Ensemble | L149-156 | L158 |
| Ensemble | L205-212 | L214 |
| Single Model (validation IC) | L285-292 | L293 |
| Single Model (backtest) | L320-327 | L329 |

```python
backtest_config = load_backtest_config()
if "backtest_start" in config.get("data", {}):
    backtest_config["period"]["start"] = config["data"]["backtest_start"]
if "backtest_end" in config.get("data", {}):
    backtest_config["period"]["end"] = config["data"]["backtest_end"]
if "h5_path" in config.get("data", {}):
    backtest_config["data"]["h5_path"] = config["data"]["h5_path"]
df, labels_df, test_dates, prices, benchmark_returns = load_backtest_data(backtest_config)
```

**해결**: 두 중복을 하나의 헬퍼로 통합

```python
def load_backtest_config_with_overrides(yaml_config):
    """Load backtest config and apply YAML overrides."""
    bt_config = load_backtest_config()
    data = yaml_config.get("data", {})
    for key, bt_key in [("backtest_start", "start"), ("backtest_end", "end")]:
        if key in data:
            bt_config["period"][bt_key] = data[key]
    if "h5_path" in data:
        bt_config["data"]["h5_path"] = data["h5_path"]
    return bt_config


def prepare_backtest_env(yaml_config):
    """Load backtest config + overrides + data in one call."""
    bt_config = load_backtest_config_with_overrides(yaml_config)
    df, labels_df, test_dates, prices, benchmark_returns = load_backtest_data(bt_config)
    return bt_config, df, labels_df, test_dates, prices, benchmark_returns
```

---

### 2. 모델 인스턴스화 + inspect 시그니처 필터링 — 3회 반복

| 위치 | 파일:라인 |
|------|----------|
| pipeline.py | L310-316 (single model backtest) |
| ensemble.py | L136-140 (ensemble predictions) |
| model_trainer.py | L157-164 (training) |

```python
_sig = inspect.signature(model_class.__init__)
_valid = set(_sig.parameters.keys()) - {"self"}
_kwargs = {k: v for k, v in model_config.items() if k in _valid}
model = model_class(**_kwargs)
```

**해결**: `data/cache.py`에 `create_model_instance(model_class, config_dict)` 헬퍼 추가

```python
def create_model_instance(model_class, config_dict):
    sig = inspect.signature(model_class.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in config_dict.items() if k in valid}
    return model_class(**kwargs)
```

---

### 3. risk_analysis 결과 → results_row 매핑 로직 — 2회 반복

| 위치 | 파일:라인 |
|------|----------|
| pipeline.py | L382-445 (single model) |
| ensemble.py | L239-301 (ensemble) |

거의 동일한 구조:
- `benchmark_returns` 여부 분기
- `risk_analysis()` 호출
- `results_row` dict 구성
- `sharpe_ratio`, `max_drawdown`, `win_rate` 계산

`_run_ensemble_backtest_and_report`로 일부 분리되어 있으나, single model 경로는 인라인으로 중복.

**해결**: 공통 `build_results_row(...)` 함수를 `evaluation/reporting.py`에 추출.

---

## 🔗 모듈 관계도

```
pipeline.py (진입점)
│
├── workflow/ensemble.py
│   ├── train_all_seeds_and_filter()
│   │   ├── data/cache.py ─── get_h5_data()
│   │   ├── model/model_trainer.py ─── ModelTrainer
│   │   └── evaluation/ic.py ─── compute_trainer_validation_ic()
│   │
│   ├── compute_ensemble_predictions()
│   │   ├── model/inference.py ─── run_inference()
│   │   └── (직접 model_class 인스턴스화 + torch.load)
│   │
│   └── _run_ensemble_backtest_and_report()
│       ├── backtest/engine.py ─── run_backtest() ×2
│       ├── evaluation/ic.py ─── compute_ic_metrics()
│       └── evaluation/risk.py ─── risk_analysis()
│
├── data/dataloader.py ─── load_backtest_config(), load_backtest_data()
├── data/cache.py ─── load_model_class()
├── model/model_trainer.py ─── ModelTrainer (학습)
├── model/inference.py ─── load_model_and_scaler(), run_inference()
├── evaluation/ic.py ─── compute_trainer_validation_ic(), compute_ic_metrics()
├── evaluation/risk.py ─── risk_analysis(), compute_daily_pred_chg(), compute_portfolio_turnover()
└── backtest/engine.py ─── run_backtest()
```

---

## 📋 우선순위 정리

| 우선순위 | 항목 | 유형 | 난이도 |
|---------|------|------|--------|
| 🔴 High | Backtest config 오버라이드 + `load_backtest_data` 묶음 중복 | 중복 | 쉬움 |
| 🔴 High | 모델 인스턴스화 3회 중복 | 중복 | 쉬움 |
| 🔴 High | `sharpe_ratio` 항상 0.0 (결과 보고서 오류 전파) | 버그 | 보통 |
| 🟡 Medium | `total_params` 캐시 무시 재로드 | 버그 | 쉬움 |
| 🟡 Medium | risk 결과 → results_row 매핑 중복 | 중복 | 보통 |
| 🟡 Medium | `load_model_and_scaler(seed=None)` 엉뚱한 파일 로드 위험 | 잠재 버그 | 쉬움 |
| 🟡 Medium | `val_ic_result` truthiness 항상 True | 버그 | 쉬움 |
| 🟢 Low | `initial_capital` 미사용 변수 | 버그 | 쉬움 |
| 🟢 Low | `load_backtest_config()` 취약한 경로 | 잠재 버그 | 쉬움 |
