#!/usr/bin/env bash
# 一键运行 BEVFusion（nuScenes mini）
#
# 用法:
#   bash run_bevfusion.sh demo [N]           # 推理前 N 帧（默认 10，最稳）
#   bash run_bevfusion.sh train-lidar        # 训练 LiDAR-only
#   bash run_bevfusion.sh train-fusion       # 训练 Camera+LiDAR（需预训练）
#   bash run_bevfusion.sh eval-lidar         # 官方评测 LiDAR（CUDA12 下可能中途失败）
#   bash run_bevfusion.sh eval-fusion        # 官方评测融合模型
#
# 环境变量:
#   EPOCHS=6 bash run_bevfusion.sh train-fusion
#   BATCH=2  bash run_bevfusion.sh train-fusion   # samples_per_gpu，建议 >=2

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion
export OMPI_MCA_plm_rsh_agent="${OMPI_MCA_plm_rsh_agent:-sh}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-tcp,self}"

MODE="${1:-demo}"
EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-2}"

LIDAR_CFG="configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml"
FUSION_CFG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

case "$MODE" in
  demo)
    bash "$REPO_ROOT/run_demo_infer.sh" "${2:-10}"
    ;;
  eval-lidar)
    test -f pretrained/lidar-only-det.pth || { echo "缺少 pretrained/lidar-only-det.pth"; exit 1; }
    torchpack dist-run -np 1 python tools/test.py \
      "$LIDAR_CFG" pretrained/lidar-only-det.pth \
      --eval bbox \
      --cfg-options data.samples_per_gpu="$BATCH" data.workers_per_gpu=2
    ;;
  train-lidar)
    echo "[INFO] LiDAR-only 训练, epochs=$EPOCHS, batch=$BATCH"
    torchpack dist-run -np 1 python tools/train.py \
      "$LIDAR_CFG" \
      --data.samples_per_gpu "$BATCH" \
      --data.workers_per_gpu 2 \
      --max_epochs "$EPOCHS"
    ;;
  eval-fusion)
    test -f pretrained/bevfusion-det.pth || { echo "缺少 pretrained/bevfusion-det.pth"; exit 1; }
    torchpack dist-run -np 1 python tools/test.py \
      "$FUSION_CFG" pretrained/bevfusion-det.pth \
      --eval bbox \
      --cfg-options data.samples_per_gpu="$BATCH" data.workers_per_gpu=2
    ;;
  train-fusion)
    test -f pretrained/swint-nuimages-pretrained.pth || { echo "缺少 camera backbone 预训练: pretrained/swint-nuimages-pretrained.pth"; exit 1; }
    test -f pretrained/lidar-only-det.pth || { echo "缺少 lidar-only 预训练: pretrained/lidar-only-det.pth"; exit 1; }
    echo "[INFO] Camera+LiDAR 融合训练, epochs=$EPOCHS, batch=$BATCH"
    torchpack dist-run -np 1 python tools/train.py \
      "$FUSION_CFG" \
      --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
      --load_from pretrained/lidar-only-det.pth \
      --data.samples_per_gpu "$BATCH" \
      --data.workers_per_gpu 2 \
      --max_epochs "$EPOCHS"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: demo | train-lidar | train-fusion | eval-lidar | eval-fusion"
    exit 1
    ;;
esac
