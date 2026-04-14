# Qlib Custom 작업환경 구축 계획서

새로운 환경(Vast.ai 등)에서 동일한 작업환경을 구축하는 절차.

---

## 1. GitHub SSH 키 설정

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# 공개키 복사 (GitHub > Settings > SSH and GPG keys > New SSH key 에 붙여넣기)
cat ~/.ssh/id_ed25519.pub

# 연결 확인
ssh -T git@github.com
# "Hi LettuceDaniel! You've successfully authenticated" 확인
```

---

## 2. GitHub 레포 Clone

```bash
cd /workspace
git clone git@github.com:LettuceDaniel/qlib_custom_dk.git
cd qlib_custom_dk
```

디렉토리 구조:
```
qlib_custom_dk/
├── init_env_settings/     # 환경 복구 스크립트
│   ├── restore_env.sh
│   └── restore_readme.md
├── RD-Agent/              # RD-Agent 팩터/모델 생성 프레임워크
├── qlib_research/         # 학습, 검증, 백테스트 코드
└── .gitignore
```

---

## 3. Miniconda + Conda 환경 구축

```bash
# Miniconda 설치
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda
rm miniconda.sh
ln -s /workspace/miniconda ~/miniconda

# Conda 환경 생성 (environment_rdagent4qlib.yml 사용)
cd /workspace/qlib_custom_dk
source /workspace/miniconda/etc/profile.d/conda.sh
conda env create -f init_env_settings/environment_rdagent4qlib.yml

# 환경 활성화
conda activate rdagent4qlib
```

**환경 정보:**
- Python 3.10
- 주요 패키지: torch 2.10, qlib(editabled), rdagent(editable), pydantic, scikit-learn, lightgbm, xgboost, catboost 등

> **참고**: yml 파일 내 pip 경로 `-e /workspace/qlib_backup`, `-e /workspace/RD-Agent`는 실제 경로에 맞게 수정 필요.

---

## 4. Qlib 데이터 준비

qlib_research/qlib_data/는 gitignore 처리되어 있으므로 별도 준비 필요:

```bash
# qlib 데이터 다운로드 (중국 시장 데이터)
python -m qlib.run.get_data qlib_data --target_dir /workspace/qlib_research/qlib_data --region cn
```

데이터 크기: ~620MB

---

## 5. Node.js + 개발도구 설치

```bash
# NVM + Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 24

# opencode (AI 코딩 어시스턴트)
npm install -g opencode-ai

# Claude Code
npm install -g @anthropic-ai/claude-code
```

**설치 버전:**
- Node.js: 22.x
- opencode: 1.4.3
- Claude Code: 2.1.x

---

## 6. Qlib Research 설정

```bash
cd /workspace/qlib_research

# editable install (필요시)
pip install -e .
```

모델 체크포인트와 대용량 데이터는 gitignore:
- `models/` (~697MB) - 학습된 모델 파일
- `models_tuned/` (~417MB) - 튜닝된 모델 파일
- `qlib_data/` (~620MB) - 시장 데이터

---

## 7. RD-Agent 설정

```bash
cd /workspace/RD-Agent

# .env 파일 생성 (API 키 등)
cp .env.example .env
# .env 편집하여 실제 API 키 입력

# editable install
pip install -e .
```

---

## 8. 환경 자동 복구 설정

```bash
# restore_env.sh를 bashrc에 등록
echo "source /workspace/qlib_custom_dk/init_env_settings/restore_env.sh" >> ~/.bashrc
source ~/.bashrc
```

restore_env.sh가 자동으로 처리하는 항목:
- Miniconda 설치 확인 및 환경 활성화
- NVM/Node.js 설치 확인
- opencode, claude 설치 확인
- tmux 설정 (마우스 지원, vi 모드)
- PATH 설정

---

## 9. 백업 및 복구

```bash
# 전체 워크스페이스 백업
ssh vast_v2 "tar czf - /workspace" > workspace_backup_$(date +%m%d).tar.gz

# 복구
tar xzf workspace_backup_XXXX.tar.gz -C /
```

---

## 10. Git 관리 가이드

코드 변경 후 GitHub에 반영:
```bash
cd /workspace/qlib_custom_dk
git add -A
git commit -m "변경 내용 설명"
git push
```

**Git 구조:**
```
GitHub: LettuceDaniel/qlib_custom_dk.git (private)
  ├── main 브랜치
  ├── RD-Agent/        (원본 RD-Agent + 커스터마이징)
  ├── qlib_research/   (학습/백테스트 코드, 모델/데이터 제외)
  └── init_env_settings/ (환경 복구 스크립트)
```

---

## 요약: 최소 설치 명령어

```bash
# 1. Clone
git clone git@github.com:LettuceDaniel/qlib_custom_dk.git /workspace/qlib_custom_dk

# 2. Conda 환경
source /workspace/miniconda/etc/profile.d/conda.sh
conda env create -f /workspace/qlib_custom_dk/init_env_settings/environment_rdagent4qlib.yml

# 3. 환경 자동설정
echo "source /workspace/qlib_custom_dk/init_env_settings/restore_env.sh" >> ~/.bashrc
source ~/.bashrc
```
