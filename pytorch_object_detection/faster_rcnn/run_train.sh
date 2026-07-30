#!/usr/bin/env bash
# 一键训练：使用 conda 环境 fasterrcnn
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-fasterrcnn}"
PYTHON="${CONDA_PYTHON:-/opt/anaconda/envs/${CONDA_ENV}/bin/python}"
CONFIG="${CONFIG:-configs/default.yaml}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets}"
EPOCHS="${EPOCHS:-}"
DOWNLOAD="${DOWNLOAD:-0}"

if [[ ! -x "$PYTHON" ]]; then
  # 回退到 conda activate
  # shellcheck disable=SC1091
  source /opt/anaconda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
  PYTHON="$(command -v python)"
fi

echo "=== Faster R-CNN 一键训练 ==="
echo "Python : $PYTHON"
echo "Config : $CONFIG"
echo "Data   : $DATA_ROOT  (期望存在 \$DATA_ROOT/VOCdevkit/VOC2007)"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

VOC_DIR="$DATA_ROOT/VOCdevkit/VOC2007"
if [[ ! -d "$VOC_DIR/JPEGImages" ]]; then
  echo "[INFO] 未找到 $VOC_DIR，开始下载 VOC2007 ..."
  DOWNLOAD=1
fi

EXTRA=()
if [[ "$DOWNLOAD" == "1" ]]; then
  "$PYTHON" download_voc.py --root "$DATA_ROOT" --year 2007
fi
if [[ -n "$EPOCHS" ]]; then
  EXTRA+=(--epochs "$EPOCHS")
fi

# 确保配置里的 data.root 与 DATA_ROOT 一致（若用户改了环境变量）
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] 启动联合训练 ..."
"$PYTHON" train.py --config "$CONFIG" "${EXTRA[@]}"

echo "[INFO] 完成。权重目录见配置 joint_output_dir（默认 outputs/joint/）"
