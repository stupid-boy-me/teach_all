#!/usr/bin/env bash
# 在 mini val 上跑前 N 帧，验证链路（避开个别样本触发的 spconv/CUDA 兼容问题）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevfusion
export OMPI_MCA_plm_rsh_agent="${OMPI_MCA_plm_rsh_agent:-sh}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-tcp,self}"

MAX_SAMPLES="${1:-10}"

python - <<PY
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader, Subset
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval

cfg_path = 'configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml'
configs.load(cfg_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=cfg_path)
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 0
cfg.model.train_cfg = None

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, 'pretrained/lidar-only-det.pth', map_location='cpu')
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

dataset = build_dataset(cfg.data.test)
max_n = min(int("${MAX_SAMPLES}"), len(dataset))
print(f'Running inference on first {max_n}/{len(dataset)} val samples...')

total_boxes = 0
class_names = cfg.object_classes
from collections import Counter
cls_counter = Counter()

for i in range(max_n):
    data = dataset[i]
    # collate like dataloader
    from mmcv.parallel import collate, scatter
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [0])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    boxes = result['boxes_3d']
    scores = result['scores_3d']
    labels = result['labels_3d']
    keep = scores > 0.3
    n = int(keep.sum())
    total_boxes += n
    for lab in labels[keep].cpu().tolist():
        cls_counter[class_names[lab]] += 1
    print(f'[{i+1:02d}/{max_n}] detections(score>0.3)={n:3d}  token={dataset.data_infos[i]["token"][:8]}...')

print('\\n==== Summary ====')
print(f'samples: {max_n}')
print(f'detections (score>0.3): {total_boxes}')
print('per-class:')
for k, v in cls_counter.most_common():
    print(f'  {k:22s} {v}')
print('DEMO OK')
PY
