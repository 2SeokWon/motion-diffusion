#!/usr/bin/env bash
# 일괄 평가 → 결과표. (일괄 평가 스크립트를 만든 뒤 채운다)
# 그 전까지는 한 구간만 수동으로 확인할 수 있다:
#   python scripts/make_control.py --bvh data/raw/Angry_FW.bvh --start_frame 3000 --waypoints 5
#   python scripts/generate.py --checkpoint_path checkpoints/<run>/best.pt --class_idx 1 --no_render
#   (start_frame은 평가 구간 = 각 파일의 뒤 15% 안에서 고른다)
set -euo pipefail
echo "TODO: 일괄 평가 스크립트 작성 후 연결" >&2
exit 1
