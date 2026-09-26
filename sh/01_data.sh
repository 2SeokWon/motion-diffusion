#!/usr/bin/env bash
# 데이터 준비: (압축 해제) → (전처리가 없으면 전처리) → 학습 구간 통계 계산 → 파일 검사
# 사용: bash sh/01_data.sh [md_data.tgz]
#   md_data.tgz는 로컬에서 `tar -czf md_data.tgz data/raw data/processed` 로 만든 파일
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
CONFIG=${CONFIG:-config.yml}
ARCHIVE=${1:-}

if [ -n "$ARCHIVE" ]; then
    echo "== 압축 해제: $ARCHIVE =="
    tar -xzf "$ARCHIVE"
fi

n_raw=$(find data/raw -maxdepth 1 -name '*.bvh' 2>/dev/null | wc -l)
if [ "$n_raw" -eq 0 ]; then
    echo "ERROR: data/raw에 BVH가 없습니다. 데이터 압축 파일을 인자로 주세요." >&2
    exit 1
fi
echo "raw BVH: $n_raw files"

if [ ! -f data/processed/metadata.json ]; then
    echo "== 전처리 (data/processed 없음) =="
    "$PY" scripts/preprocess.py
fi

echo "== 학습 구간 정규화 통계 =="
"$PY" scripts/split_stats.py --config "$CONFIG"

echo "== 검사 =="
"$PY" - "$CONFIG" <<'EOF'
import sys, os, json
import numpy as np
sys.path.insert(0, '.')
from core.config import load_config
cfg = load_config(sys.argv[1])
P = cfg.data.processed_dir
meta = json.load(open(os.path.join(P, 'metadata.json')))
meta = [c for sub in meta for c in sub] if meta and isinstance(meta[0], list) else meta
missing = [c['path'] for c in meta if not os.path.exists(os.path.join(P, c['path']))]
assert not missing, f"missing clips: {missing}"
for name in ['root_pos', 'position', 'rotation', 'foot', 'abs_traj']:
    for kind in ['mean', 'std']:
        a = np.load(os.path.join(cfg.data.stats_dir, f'{name}_{kind}.npy'))
        assert np.isfinite(a).all(), f"{name}_{kind} has NaN/Inf"
assert os.path.exists(cfg.generation.skeleton_template), cfg.generation.skeleton_template
print(f"OK: {len(meta)} clips, {sum(c['length'] for c in meta)} frames, stats in {cfg.data.stats_dir}")
EOF
