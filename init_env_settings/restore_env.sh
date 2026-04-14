#!/bin/bash

# ===== 의존성 체크 및 설치 =====
if [ ! -d "/workspace/miniconda" ]; then
    echo "🔧 Miniconda 설치 중..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda
    rm miniconda.sh
fi

if [ ! -L "$HOME/miniconda" ]; then
    ln -s /workspace/miniconda "$HOME/miniconda"
fi

if [ ! -d "/opt/nvm" ]; then
    echo "🔧 NVM 및 Node.js 설치 중..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install 24
    mkdir -p /opt/nvm
    ln -s ~/.nvm /opt/nvm
fi

# ===== Tmux 설정 =====
touch ~/.no_auto_tmux

if ! command -v xsel &> /dev/null; then
    echo "🔧 xsel 설치 중..."
    sudo apt install -y xsel
fi

if [ ! -d "/opt/nvm" ]; then
    if ! grep -q "set -g mouse on" ~/.tmux.conf 2>/dev/null; then
        cat >> ~/.tmux.conf << 'EOF'
set -g mouse on
setw -g mode-keys vi
bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "xsel -in -selection clipboard"
EOF
    fi
fi

# ===== opencode와 claude 설치 체크 =====
if ! command -v opencode &> /dev/null; then
    echo "🔧 opencode 설치 중..."
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    npm install -g opencode-ai
fi

if ! command -v claude &> /dev/null; then
    echo "🔧 claude 설치 중..."
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    npm install -g @anthropic-ai/claude-code
fi

# ===== 공통 환경 설정 =====
source /workspace/miniconda/etc/profile.d/conda.sh
export PATH="/workspace/miniconda/envs/rdagent4qlib/bin:$PATH"
export XDG_CONFIG_HOME="/workspace/.config"

# ===== 환경만 설정 (터미널 열 때 자동 실행) =====
if [ -z "$RDAGENT_ENV_INITIALIZED" ]; then
    export RDAGENT_ENV_INITIALIZED=1
    conda activate rdagent4qlib
fi
