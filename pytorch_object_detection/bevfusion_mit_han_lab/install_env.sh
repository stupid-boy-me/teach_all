#!/usr/bin/env bash
# BEVFusion 一键安装脚本
# 用法:
#   bash install_env.sh
#   bash install_env.sh --with-data          # 额外做数据软链接 + create_data
#   bash install_env.sh --skip-compile       # 跳过 python setup.py develop
#   bash install_env.sh --force              # 已存在 bevfusion 环境也重建

set -euo pipefail

ENV_NAME="bevfusion"
PYTHON_VERSION="3.8"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUSCENES_SRC="${NUSCENES_SRC:-/workspace/datasets/v1.0-mini}"

WITH_DATA=0
SKIP_COMPILE=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --with-data) WITH_DATA=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    --force) FORCE=1 ;;
    -h|--help)
      echo "Usage: bash install_env.sh [--with-data] [--skip-compile] [--force]"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $arg"
      exit 1
      ;;
  esac
done

log()  { echo -e "\n\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\n\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# ---------------------------------------------------------------------------
# 0) 基础检查
# ---------------------------------------------------------------------------
log "检查 conda / nvidia-smi"
command -v conda >/dev/null 2>&1 || err "未找到 conda，请先安装 Anaconda/Miniconda"
command -v nvidia-smi >/dev/null 2>&1 || warn "未检测到 nvidia-smi，后续 CUDA 扩展可能失败"

# 让当前 shell 能 conda activate
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1) 创建/重建环境
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$FORCE" -eq 1 ]]; then
    log "删除已有环境: $ENV_NAME"
    conda deactivate >/dev/null 2>&1 || true
    conda env remove -n "$ENV_NAME" -y
  else
    log "环境 $ENV_NAME 已存在，将复用（如需重建请加 --force）"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "创建 conda 环境: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
log "当前 Python: $(which python) ($(python -V 2>&1))"

# ---------------------------------------------------------------------------
# 2) PyTorch
# ---------------------------------------------------------------------------
log "安装 PyTorch 1.10.1 + CUDA 11.3"
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

# ---------------------------------------------------------------------------
# 3) 基础 pip 包
# ---------------------------------------------------------------------------
log "安装 Pillow / tqdm / torchpack"
pip install Pillow==8.4.0 tqdm torchpack
# mmcv 1.4.0 的 pretty_text 需要旧版 yapf（新版去掉了 verify 参数）
pip install 'yapf==0.32.0'
# torch1.10 tensorboard 需要旧 setuptools（否则 distutils.version 缺失）
pip install 'setuptools==59.5.0' 'tensorboard==2.14.0'

# ---------------------------------------------------------------------------
# 4) mmcv-full / mmdet / nuscenes
# ---------------------------------------------------------------------------
log "安装 mmcv-full==1.4.0（OpenMMLab 预编译轮子）"
pip uninstall -y mmcv mmcv-full >/dev/null 2>&1 || true
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

log "安装 mmdet==2.20.0 与 nuscenes-devkit"
pip install mmdet==2.20.0
pip install nuscenes-devkit

# ---------------------------------------------------------------------------
# 5) OpenMPI + mpi4py
# ---------------------------------------------------------------------------
log "安装 openmpi / mpi4py / openssh (conda-forge)"
conda install -y -c conda-forge "openmpi=4.1.*" "mpi4py=3.1.*" openssh

# ---------------------------------------------------------------------------
# 6) numba
# ---------------------------------------------------------------------------
log "安装 numba==0.58.1"
pip install numba==0.58.1

# ---------------------------------------------------------------------------
# 7) MPI 运行时环境变量（写入环境激活脚本，一劳永逸）
# ---------------------------------------------------------------------------
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/bevfusion_mpi.sh" <<'EOF'
export OMPI_MCA_plm_rsh_agent="${OMPI_MCA_plm_rsh_agent:-sh}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-tcp,self}"
EOF

cat > "$DEACTIVATE_DIR/bevfusion_mpi.sh" <<'EOF'
unset OMPI_MCA_plm_rsh_agent
unset OMPI_MCA_btl
EOF

# 当前 shell 立即生效
export OMPI_MCA_plm_rsh_agent=sh
export OMPI_MCA_btl=tcp,self
log "已写入 conda activate 钩子: OMPI_MCA_plm_rsh_agent / OMPI_MCA_btl"

# ---------------------------------------------------------------------------
# 8) 编译本仓库
# ---------------------------------------------------------------------------
if [[ "$SKIP_COMPILE" -eq 1 ]]; then
  warn "跳过 python setup.py develop (--skip-compile)"
else
  log "编译安装本仓库 CUDA 扩展: python setup.py develop"
  cd "$REPO_ROOT"
  export FORCE_CUDA="${FORCE_CUDA:-1}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

  # PyTorch 1.10(cu113) + 系统 CUDA 12.x 时，放宽版本检查
  EXT_PY="$CONDA_PREFIX/lib/python3.8/site-packages/torch/utils/cpp_extension.py"
  if [[ -f "$EXT_PY" ]] && grep -q 'raise RuntimeError(CUDA_MISMATCH_MESSAGE.format' "$EXT_PY"; then
    cp -n "$EXT_PY" "${EXT_PY}.bak" 2>/dev/null || true
    python - <<'PY'
from pathlib import Path
import os
p = Path(os.environ['CONDA_PREFIX']) / 'lib/python3.8/site-packages/torch/utils/cpp_extension.py'
text = p.read_text()
old = 'raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))'
new = 'print(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda)); print("[WARN] bypass CUDA version mismatch check")'
if old in text:
    p.write_text(text.replace(old, new, 1))
    print('patched torch cpp_extension CUDA check')
PY
  fi

  # CUDA12 + 旧 PyTorch：Macros.h 里 uint32_t 未定义
  MACROS="$CONDA_PREFIX/lib/python3.8/site-packages/torch/include/c10/macros/Macros.h"
  if [[ -f "$MACROS" ]] && grep -q 'constexpr uint32_t CUDA_MAX_THREADS_PER_SM' "$MACROS"; then
    cp -n "$MACROS" "${MACROS}.bak" 2>/dev/null || true
    sed -i 's/constexpr uint32_t CUDA_MAX_THREADS_PER_SM/constexpr unsigned int CUDA_MAX_THREADS_PER_SM/g' "$MACROS"
    sed -i 's/constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK/constexpr unsigned int CUDA_MAX_THREADS_PER_BLOCK/g' "$MACROS"
    sed -i 's/constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK/constexpr unsigned int CUDA_THREADS_PER_BLOCK_FALLBACK/g' "$MACROS"
    log "patched torch Macros.h uint32_t -> unsigned int"
  fi

  python setup.py develop
fi

# ---------------------------------------------------------------------------
# 9) 可选：数据准备
# ---------------------------------------------------------------------------
if [[ "$WITH_DATA" -eq 1 ]]; then
  log "准备 nuScenes 数据软链接与 infos"
  if [[ ! -d "$NUSCENES_SRC" ]]; then
    err "数据目录不存在: $NUSCENES_SRC
可用环境变量覆盖，例如:
  NUSCENES_SRC=/your/path/v1.0-mini bash install_env.sh --with-data"
  fi
  mkdir -p "$REPO_ROOT/data"
  ln -sfn "$NUSCENES_SRC" "$REPO_ROOT/data/nuscenes"
  ls -la "$REPO_ROOT/data/nuscenes" | head
  python "$REPO_ROOT/tools/create_data.py" nuscenes \
    --root-path "$REPO_ROOT/data/nuscenes" \
    --out-dir "$REPO_ROOT/data/nuscenes" \
    --extra-tag nuscenes \
    --version v1.0-mini \
    --max-sweeps 10
fi

# ---------------------------------------------------------------------------
# 10) 自检
# ---------------------------------------------------------------------------
log "运行环境自检"
python - <<'PY'
import torch, mmcv, mmdet, numba, torchpack, PIL
from mpi4py import MPI
from mmcv.ops import box_iou_rotated
import nuscenes
print('torch     ', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('mmcv      ', mmcv.__version__)
print('mmdet     ', mmdet.__version__)
print('numba     ', numba.__version__)
print('Pillow    ', PIL.__version__)
print('torchpack ', torchpack.__version__)
print('mpi4py    ', 'OK size=', MPI.COMM_WORLD.Get_size())
print('nuscenes  ', 'OK')
print('mmcv ops  ', 'OK')
try:
    import mmdet3d
    print('mmdet3d   ', mmdet3d.__file__)
except Exception as e:
    print('mmdet3d   ', 'NOT READY:', e)
    raise SystemExit(1)
print('ALL CHECKS PASSED')
PY

log "安装完成"
echo "------------------------------------------------------------"
echo "下次使用:"
echo "  conda activate $ENV_NAME"
echo "  cd $REPO_ROOT"
echo
echo "可选训练（LiDAR-only, 单卡）:"
echo "  torchpack dist-run -np 1 python tools/train.py \\"
echo "    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \\"
echo "    --data.samples_per_gpu 1"
echo
if [[ "$WITH_DATA" -eq 0 ]]; then
  echo "若还没准备数据，可再执行:"
  echo "  NUSCENES_SRC=$NUSCENES_SRC bash install_env.sh --with-data --skip-compile"
fi
echo "------------------------------------------------------------"
