#!/usr/bin/env bash
# 用法: ./train.sh
# 可选: ./train.sh --epochs 6
set -e
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source /opt/anaconda/etc/profile.d/conda.sh
conda activate fasterrcnn

python train.py --config configs/default.yaml "$@"
