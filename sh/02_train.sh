#!/usr/bin/env bash
# 학습을 백그라운드(nohup)로 시작한다. SSH 연결이 끊겨도 계속 돈다.
# 사용:
#   bash sh/02_train.sh                                         # 새로 학습
#   RESUME=checkpoints/<run>/last.pt bash sh/02_train.sh        # 이어서 학습
#   CONFIG=my.yml bash sh/02_train.sh                           # 다른 설정
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
CONFIG=${CONFIG:-config.yml}

if [ ! -f data/processed/stats_train/abs_traj_mean.npy ]; then
    echo "ERROR: 학습 통계가 없습니다. 먼저 bash sh/01_data.sh 를 실행하세요." >&2
    exit 1
fi
if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
    echo "ERROR: 이미 학습 중입니다 (PID $(cat logs/train.pid)). 멈추려면: kill \$(cat logs/train.pid)" >&2
    exit 1
fi

ARGS=(--config "$CONFIG")
if [ -n "${RESUME:-}" ]; then
    ARGS+=(--resume "$RESUME")
fi

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
nohup "$PY" -u scripts/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > logs/train.pid

echo "학습 시작: PID $(cat logs/train.pid)"
echo "  로그 보기:  tail -f $LOG"
echo "  평가 손실:  grep 'eval loss' $LOG"
echo "  멈추기:     kill \$(cat logs/train.pid)"
echo "  이어서:     RESUME=checkpoints/<run>/last.pt bash sh/02_train.sh"
