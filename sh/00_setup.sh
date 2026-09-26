#!/usr/bin/env bash
# 서버 환경 준비: GPU/PyTorch 확인 → 필요한 패키지 설치 → wandb 로그인 확인
# PyTorch + CUDA가 설치된 서버 이미지를 전제로 한다 (PyTorch는 새로 설치하지 않는다).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARN: nvidia-smi not found"

echo "== PyTorch =="
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch가 없습니다. PyTorch + CUDA 이미지로 서버를 만들거나 pip로 설치하세요." >&2
    exit 1
fi
"$PY" - <<'EOF'
import sys, torch
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda} | available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA를 쓸 수 없습니다. GPU 서버/이미지를 확인하세요.")
print(f"device: {torch.cuda.get_device_name(0)}")
EOF

echo "== Packages =="
"$PY" -m pip install -q -r requirements-server.txt
"$PY" -c "import numpy, yaml, tqdm, joblib, wandb; from pyglm import glm; print('packages OK')"

echo "== wandb =="
if [ -n "${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" ~/.netrc; then
    echo "wandb: 로그인됨"
elif [ "${WANDB_MODE:-}" = "offline" ] || [ "${WANDB_MODE:-}" = "disabled" ]; then
    echo "wandb: WANDB_MODE=${WANDB_MODE}"
else
    echo "wandb: 로그인 필요 → 'wandb login' 실행 (또는 export WANDB_MODE=offline)"
    exit 1
fi
