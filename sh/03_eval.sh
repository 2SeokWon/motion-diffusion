#!/usr/bin/env bash
# 일괄 평가 → 결과표 (results/eval_<run>_<ckpt>_<sampler>_<시간>/summary.md)
# 평가 창: 파일마다 학습에 안 쓴 뒤 15%에서 겹치지 않는 180프레임 창 2개 (49파일 → 98개)
# 조건:    경유점 goal / 2 / 5 / 10 / dense
# 사용:
#   bash sh/03_eval.sh checkpoints/<run>/best.pt            # DDPM 1000 + DDIM 50/20/10
#   SAMPLERS="ddim50" bash sh/03_eval.sh checkpoints/<run>/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

CKPT=${1:?사용법: bash sh/03_eval.sh checkpoints/<run>/best.pt}
SAMPLERS=${SAMPLERS:-"ddpm ddim50 ddim20 ddim10"}

mkdir -p logs
for S in $SAMPLERS; do
    if [ "$S" = "ddpm" ]; then
        ARGS=(--sampler ddpm)
    else
        ARGS=(--sampler ddim --steps "${S#ddim}")
    fi
    LOG="logs/eval_${S}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== $S → $LOG"
    "$PY" -u scripts/evaluate.py --checkpoint_path "$CKPT" "${ARGS[@]}" 2>&1 | tee "$LOG"
done

echo
echo "결과표: ls -d results/eval_*  (각 폴더의 summary.md)"
echo "[로컬 PC로 가져오기] scp -r <user>@<server>:~/motion-diffusion/results/eval_* results/"
