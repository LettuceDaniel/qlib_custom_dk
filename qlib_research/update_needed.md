# 🔍 `train_and_backtest.py` 종합 분석 결과

> 최종 업데이트: 2026-04-11
> 대상 파일: `train_valid_backtest/train_and_backtest.py` (2,170줄)

---

## 📊 전체 현황 요약

| 상태 | 건수 | 항목 |
|------|------|------|
| ✅ 수정 완료 | 8건 | #1, #2, #3, #4, #6, #7, #8, #10 |
| ✅ 확인 결과 문제없음 | 4건 | #5, #14, #15, #17 |
| ⚠️ 위험도 낮아 보류 | 1건 | #9 |
| 🔶 이미 해결됨 (코드 변경 반영) | 2건 | #13, #16 |
| ⏳ 예정 | 1건 | #11 (모듈 분리) |

---

## ✅ 수정 완료 이력

### ~~1. `execute_buy` 반환값 타입 불일치 (버그)~~ ✅ 수정 완료

**수정 내용**: `return False` → `return 0.0`으로 통일. `execute_buy`는 더 이상 포지션을 직접 추가하지 않고 비용만 계산.

### ~~2. 매수 실패 시 포지션 잔존 가능 (버그)~~ ✅ 수정 완료

**수정 내용**: `execute_buy`에서 `position[stock_id] = {...}` 제거. caller(`run_backtest`)에서 `available >= total_needed` 확인 후 포지션 추가. 실패 시 롤백 불필요한 원자적 구조로 변경.

### ~~3. `torch.load` 보안 경고 (PyTorch 2.6+에서 에러 발생 가능)~~ ✅ 수정 완료

**수정 내용**: `train_and_backtest.py` 3개소 + `model_architecture.py` 1개소, 총 4개소에 `weights_only=False` 추가. PyTorch 2.10.0 환경에서 호환성 확보.

### ~~4. H5 데이터 중복 로딩 (메모리 낭비 + 성능 저하)~~ ✅ 수정 완료

**수정 내용**: 글로벌 캐시 `_h5_cache` + `get_h5_data()` 도입. 4개 호출지점 모두 캐시 사용. 프로세스당 최초 1회만 디스크 I/O (기존 최대 17회 → 1회).

### ~~6. 수익률 ffill + fillna(0)은 금융적으로 부정확~~ ✅ 수정 완료

**수정 내용**: `ffill()` 및 `fillna(0)` 제거. NaN 수익률은 그대로 유지하며, `apply_returns()`에서 NaN 종목을 포지션에서 자동 제거.

### ~~7. IC 계산 로직 중복 (DRY 위반)~~ ✅ 수정 완료

**수정 내용**: 공통 `_calc_ic(predictions, labels_df, dates, min_obs=10)` 추출. `compute_ic_metrics`와 `compute_trainer_validation_ic`의 중복 IC/ICIR/RankIC 계산을 단일 함수로 통합.

### ~~8. `qlib.init()` 중복 호출~~ ✅ 수정 완료

**수정 내용**: `qlib.init()`을 `main()` 모드 분기 전 1곳으로 통합. `train_all_seeds_and_filter()` 및 각 모드 내부의 3곳 호출 제거.

### ~~10. 백테스트 과도한 매매 시 현금 음수 가능~~ ✅ 수정 완료

**수정 내용**: `value_per` 계산 시 수수료를 사전 반영 — `investable / (1 + open_cost)`로 할당액 역산하여 `total_needed <= investable` 보장.

---

## ✅ 확인 결과 문제없음

### ~~5. 시드 재현성 불완전 — numpy/random 미설정~~ ✅ 문제없음

**확인 내용**: 27개 `model_architecture.py` 모두 `nn.init.xavier_uniform_` 사용 (수학 공식 기반, 시드 무관). `np.random`/`random` 사용 0건. `torch.manual_seed` + DataLoader `generator`로 이미 충분히 통제됨.

### ~~14. `load_backtest_config` 경로 하드코딩~~ ✅ 수정 불필요

**확인 내용**: `__file__` 기반 2단계 상위 경로는 프로젝트 루트를 찾는 Python 표준 패턴. `env.toml`은 모델이 아닌 프로젝트 공통 백테스트 설정이므로 CLI 인자 분리 불필요. 심볼릭 링크도 `os.path.abspath`가 resolve.

### ~~15. stdout 리다이렉트 방식의 로깅~~ ✅ 수정 불필요

**확인 내용**: `try/finally`로 학습 직후 즉시 복원. 싱글스레드 환경이므로 부작용 없음. 병렬 학습은 별도 프로세스라 `sys.stdout` 독립. `trainer.train()`이 `print()` 기반이라 stdout 교체가 가장 실용적.

### ~~17. `toml` 임포트~~ ✅ 이미 해결됨

**확인 내용**: 현재 코드가 이미 `try: import tomllib as toml / except: import toml` 패턴을 사용 중. Python 3.10 환경에서는 `tomllib` 미지원이므로 `toml` fallback이 올바름.

---

## 🔶 이미 해결됨 (코드 변경으로 자연스럽게 해결)

### ~~13. `HDF5DataLoader._data` 인스턴스 캐시~~ 🔶 글로벌 캐시로 사실상 해결

**현황**: #4에서 `_h5_cache` 글로벌 캐시를 도입하면서 `self._data`는 글로벌 캐시의 copy를 한 번 더 저장하는 2차 캐시 역할. 메모리 누수는 아니며(같은 프로세스 내에서만 유지), 향후 모듈 분리 시 `self._data`를 제거하고 매번 `get_h5_data()`를 직접 호출하는 것으로 정리 가능.

### ~~16. `importlib.util` 모델 로딩 중복~~ ✅ 이미 해결됨

**현황**: `load_model_class(model_folder)` 함수로 통합 + `_model_class_cache` 캐싱까지 구현됨. 현재 `importlib` 호출은 1곳(L182)만 존재.

---

## ⚠️ 보류

### 9. `results_combined.csv` 병렬 쓰기 시 레이스 컨디션

**현황**: 여러 프로세스가 동시에 read-modify-write 수행. 학습 시간이 달라 CSV write 타이밍이 분리되므로 **실제 트리거 확률 극히 낮음**. 모델 파일(`model_seed{N}.pt`)은 seed별 분리되어 영향 없음.
**필요시 해결안**: seed별 개별 파일(`results_seed{N}.csv`)로 분리.

---

## ⏳ 예정

### 11. 단일 파일 → 모듈 분리

`structure` 파일에 분리 계획이 명시되어 있음:

```
Section 1 → pipeline/data.py
Section 2 → pipeline/trainer.py
Section 3 → pipeline/inference.py
Section 4 → pipeline/backtest.py
Section 5 → pipeline/metrics.py
Section 6 → run.py (CLI)
```

---

## 🟡 참고 — 추가로 확인한 항목

### 12. `import qlib` 모듈 레벨

**현황**: L157에 `import qlib` 존재. `qlib.init()`은 런타임에 호출되며, `import qlib` 자체는 사이드이펙트 없음. 단위 테스트 시 qlib 의존성이 강제되는 건 사실이나, 이 프로젝트의 성격(qlib 없이는 동작하지 않는 퀀트 파이프라인)상 실제 문제는 아님. #11 모듈 분리 시 자연스럽게 정리 가능.
