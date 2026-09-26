#!/usr/bin/env bash
# 서버를 빌린 직후 한 번에: 환경 준비 → 데이터 준비 → 학습 시작(백그라운드)
#
# [로컬 PC] 데이터 묶어서 보내기 (data/는 git에 없음)
#   tar -czf md_data.tgz data/raw data/processed
#   scp md_data.tgz <user>@<server>:~/
#
# [서버]
#   git clone -b feature/waypoint-cond https://github.com/2SeokWon/motion-diffusion.git
#   cd motion-diffusion
#   wandb login                      # 처음 한 번 (또는 export WANDB_MODE=offline)
#   bash sh/run_all.sh ~/md_data.tgz
#   tail -f logs/train_*.log
#
# [결과 가져오기, 로컬 PC에서]
#   scp -r <user>@<server>:~/motion-diffusion/checkpoints/<run> checkpoints/
set -euo pipefail
cd "$(dirname "$0")/.."

bash sh/00_setup.sh
bash sh/01_data.sh "${1:-}"
bash sh/02_train.sh
