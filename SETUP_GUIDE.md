# Qlib Custom GitHub 레포 구축 가이드

이 가이드는 GitHub 레포를 처음부터 구축하는 방법을 설명합니다.
레포는 이미 GitHub에서 만들었다고 가정합니다.

---

## 레포 구조

```
qlib_custom_dk/              ← 여기에 .git (git init)
├── .gitignore
├── init_env_settings/       # 추적됨 (환경 복구 스크립트)
│   ├── restore_env.sh
│   ├── restore_readme.md
│   └── environment_rdagent4qlib.yml
├── RD-Agent/                # 추적됨 (소스코드만)
│   ├── rdagent/             # 추적됨
│   ├── test/                # 추적됨
│   ├── docs/                # 추적됨
│   ├── ...
│   ├── pickle_cache/        # ignore됨 (282MB)
│   ├── git_ignore_folder/   # ignore됨 (3.4MB)
│   ├── qlib_backup/qlib_data/    # ignore됨
│   ├── qlib_backup/pickle_cache/ # ignore됨
│   ├── qlib_backup/git_ignore_folder/ # ignore됨
│   ├── qlib_backup/log/          # ignore됨
│   ├── log/                     # ignore됨
│   ├── .env                     # ignore됨
│   ├── prompt_cache.db          # ignore됨 (31MB)
│   └── *.db, *.pkl, *.pth       # ignore됨
└── qlib_research/           # 추적됨 (소스코드만)
    ├── train_valid_backtest/    # 추적됨
    ├── qlib/                    # 추적됨
    ├── config/                  # 추적됨
    ├── models/                  # ignore됨 (697MB - 모델 체크포인트)
    ├── models_tuned/            # ignore됨 (417MB - 튜닝된 모델)
    ├── qlib_data/               # ignore됨 (620MB - 시장 데이터)
    └── __pycache__/             # ignore됨
```

---

## 구축 순서

### 1. 빈 디렉토리에 git init

```bash
mkdir qlib_custom_dk
cd qlib_custom_dk
git init
git branch -m main
```

### 2. .gitignore 작성

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Virtual environments
.env
.venv
venv/

# IDE
.vscode/
.idea/
.cursor/
.claude/

# OS
.DS_Store
*.swp

# Caches
.cache/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*cache*/

# DB / binary / model files
*.db
*.pkl
*.h5
*.pth
*.bin

# Logs
log/

# qlib_research: 모델 체크포인트와 시장 데이터
qlib_research/models/
qlib_research/models_tuned/
qlib_research/qlib_data/

# RD-Agent: 생성된 파일과 캐시
RD-Agent/pickle_cache/
RD-Agent/git_ignore_folder/
RD-Agent/prompt_cache.db
RD-Agent/mlruns/
RD-Agent/reports/
RD-Agent/factor_template/mlruns/
RD-Agent/qlib_backup/qlib_data/
RD-Agent/qlib_backup/pickle_cache/
RD-Agent/qlib_backup/git_ignore_folder/
RD-Agent/qlib_backup/log/

*.out
EOF
```

### 3. 소스코드 복사

기존 작업 환경에서 git으로 추적되던 파일만 복사합니다.
각 원본 폴더에 이미 .git이 있으므로 `git ls-files`로 추적 파일 목록을 가져옵니다.

```bash
# 원본 위치 (기존 작업환경)
ORIG_RDA=/workspace/RD-Agent
ORIG_QLIB=/workspace/qlib_research

# RD-Agent 복사 (git 추적 파일만)
cd $ORIG_RDA
git ls-files | rsync -a --files-from=- . /path/to/qlib_custom_dk/RD-Agent/

# qlib_research 복사 (git 추적 파일만)
cd $ORIG_QLIB
git ls-files | rsync -a --files-from=- . /path/to/qlib_custom_dk/qlib_research/

# init_env_settings 복사
cp -r /workspace/init_env_settings/ /path/to/qlib_custom_dk/init_env_settings/
```

### 4. 커밋 & 푸시

```bash
cd /path/to/qlib_custom_dk
git add -A
git commit -m "Initial commit"
git remote add origin git@github.com:LettuceDaniel/qlib_custom_dk.git
git push -u origin main
```

---

## Ignore 항목 요약

| 폴더 | 크기 | 이유 |
|------|------|------|
| `qlib_research/models/` | 697MB | 모델 체크포인트 (.pt, .pth) |
| `qlib_research/models_tuned/` | 417MB | 튜닝된 모델 체크포인트 |
| `qlib_research/qlib_data/` | 620MB | qlib 시장 데이터 (.bin) |
| `RD-Agent/pickle_cache/` | 282MB | RD-Agent 실행 캐시 |
| `RD-Agent/prompt_cache.db` | 31MB | 프롬프트 캐시 DB |
| `RD-Agent/git_ignore_folder/` | 3.4MB | RD-Agent 작업 공간 |
| `RD-Agent/log/` | 177KB | 실행 로그 |
| `RD-Agent/.env` | - | API 키 등 민감 정보 |
| `RD-Agent/qlib_backup/qlib_data/` | - | 백업된 qlib 데이터 |
| `__pycache__/`, `*.pyc` | - | Python 컴파일 캐시 |
| `*.egg-info/` | - | 패키지 빌드 정보 |

**총 ignore 크기: ~2GB+**
**실제 추적(업로드) 크기: ~15MB (소스코드만)**
