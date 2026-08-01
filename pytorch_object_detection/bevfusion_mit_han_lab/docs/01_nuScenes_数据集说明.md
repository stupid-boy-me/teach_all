# nuScenes 数据集说明（面向 BEVFusion 3D 检测）

本文面向初学者，说明什么是 nuScenes、mini 版里有什么、以及它如何接到本仓库的 BEVFusion 训练流程。

---

## 1. 一句话理解 nuScenes

nuScenes 是自动驾驶常用的**多传感器**公开数据集。每一帧关键时刻）同时提供：

- **6 路环视相机**图像（语义丰富）
- **1 路顶置激光雷达（LiDAR）**点云（几何精确）
- **5 路毫米波雷达**（本仓库检测主流程默认不用）
- **3D 标注框**（位置、尺寸、朝向、类别）
- **传感器标定与车辆位姿**（把相机/雷达坐标统一到车体或 LiDAR 坐标）

BEVFusion 做 **3D 车辆/目标检测**时，核心就是：用相机 + LiDAR，在鸟瞰图（BEV）空间里预测 3D 框。

---

## 2. 你当前用的是 mini 版

| 项目 | 完整 trainval | **v1.0-mini（你现在的）** |
|------|---------------|---------------------------|
| 场景数 | 850（train）+ 150（val） | **10** |
| 关键帧 sample 数 | 约 28k / 6k | **404** |
| 用途 | 正式训练与刷榜 | 调试流程、理解数据、小规模试跑 |
| 磁盘占用 | 数百 GB 量级 | 约 **5GB+**（本机约 samples 0.7G + sweeps 4.4G） |

本机路径对应关系（与项目统一目录约定一致）：

```text
Windows:  E:\WSL\wsl_datasets\v1.0-mini
容器内:   /workspace/datasets/v1.0-mini
```

mini 已包含官方拆好的 10 个场景，例如：`scene-0061`、`scene-0103`、…、`scene-1100`。

---

## 3. 目录结构（你磁盘上已有的内容）

正确的 nuScenes 根目录应长这样（你的 mini 已经符合）：

```text
v1.0-mini/                          # 数据集根目录
├── maps/                           # 地图栅格（分割任务会用到）
├── samples/                        # 关键帧传感器数据（与标注对齐的关键帧）
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/                  # 关键帧点云 (.pcd.bin)
│   └── RADAR_*                     # 雷达（检测主配置可忽略）
├── sweeps/                         # 非关键帧的中间帧（多帧点云/图像拼接用）
│   ├── CAM_* /
│   ├── LIDAR_TOP/
│   └── RADAR_* /
└── v1.0-mini/                      # JSON 元数据（版本名与文件夹同名）
    ├── sample.json                 # 关键帧列表
    ├── sample_data.json            # 每条传感器数据的路径与时间戳
    ├── sample_annotation.json      # 3D 框标注
    ├── calibrated_sensor.json      # 内外参
    ├── ego_pose.json               # 车体位姿
    ├── scene.json / category.json / instance.json / ...
    └── ...
```

### 3.1 `samples` vs `sweeps`

- **samples**：带 3D 标注的关键时刻（keyframe），约每 0.5s 一帧。
- **sweeps**：关键帧之间的中间采集。训练时常把多帧 LiDAR sweeps 叠加到当前帧，增加点密度。

本仓库默认检测配置会加载当前帧 LiDAR + 若干历史 sweeps（见 `configs/nuscenes/default.yaml` 中的 `LoadPointsFromMultiSweeps`）。

### 3.2 JSON 元数据在干什么

可以把 `v1.0-mini/*.json` 理解成数据库表：

| 文件 | 作用 |
|------|------|
| `scene.json` | 一条连续驾驶片段 |
| `sample.json` | 场景中的关键帧 |
| `sample_data.json` | 某传感器某时刻的文件路径 |
| `sample_annotation.json` | 某个目标在某关键帧的 3D 框 |
| `calibrated_sensor.json` | 相机内参、外参（传感器→车体） |
| `ego_pose.json` | 车体在世界坐标系中的位姿 |
| `category.json` / `instance.json` | 类别与目标实例 ID |

训练代码**不会**每次从这些 JSON 现算，而是先用 `tools/create_data.py` 生成 `*.pkl` 索引，加速读取。

---

## 4. 检测关注的 10 个类别

官方细分类别很多；BEVFusion / MMDetection3D 评测常用的 **10 类**为：

1. `car`（轿车）
2. `truck`
3. `construction_vehicle`
4. `bus`
5. `trailer`
6. `barrier`
7. `motorcycle`
8. `bicycle`
9. `pedestrian`
10. `traffic_cone`

若你只关心「车辆检测」，实践中通常仍先按这 10 类跑通官方流程，再在配置里把 `object_classes` 收窄到车辆相关类（如 `car/truck/bus/...`）。**先跑通全类再改类，排错更简单。**

每个 3D 框通常包含：中心 `(x, y, z)`、尺寸 `(l, w, h)`、朝向 yaw、类别、速度等（具体字段在 info pkl / 标注里）。

---

## 5. 坐标系直觉（读代码时很有用）

常见变换链：

```text
相机像素 ↔ 相机坐标  ——内参——
相机坐标 ↔ 车体(ego) ——外参——
LiDAR 坐标 ↔ 车体(ego)
世界坐标 ↔ 车体(ego) ——ego_pose——
```

BEVFusion 把相机特征「抬」到 **BEV（鸟瞰）**，与 LiDAR 体素特征在同一俯视平面融合，所以标定矩阵（`camera2lidar`、`lidar2image` 等）必须正确。这些矩阵会在数据预处理阶段写进每个 sample 的 meta 信息。

---

## 6. 接到本仓库时需要的目录形态

官方 README 期望：

```text
bevfusion_mit_han_lab/
└── data/
    └── nuscenes/          # 软链接或拷贝到真实数据根
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-mini/     # mini 用这个；完整集则是 v1.0-trainval / v1.0-test
        ├── nuscenes_infos_train.pkl   # create_data 生成
        ├── nuscenes_infos_val.pkl
        ├── nuscenes_dbinfos_train.pkl
        └── nuscenes_database/         # GT 采样数据库（训练增强用）
```

对本机推荐做法（环境就绪后执行）：

```bash
cd /workspace/project/codes/bevfusion_mit_han_lab
mkdir -p data
ln -sfn /workspace/datasets/v1.0-mini data/nuscenes
```

然后生成索引（**必须用本仓库的 converter**，不要直接拿别的 MMDet3D 版本生成的 pkl）：

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --max-sweeps 10
```

`create_data.py` 对 `v1.0-mini` 有专门分支：会按官方 `mini_train` / `mini_val` 划分生成 train/val infos，并构建 GT database。

---

## 7. mini 版使用注意

1. **样本太少**：不适合复现论文指标；适合验证「环境 + 数据 + 训练/评测脚本」是否通。
2. **显存**：RTX 3060 12GB 跑完整 Camera+LiDAR 融合时，建议把 `samples_per_gpu` 降到 1～2。
3. **地图扩展包**：若只做 3D 目标检测，现有 `maps/` 一般够用；若做 BEV 地图分割且缺图，再按官网下载 map expansion。
4. **路径**：配置里默认 `dataset_root: data/nuscenes/`，因此软链接名必须是 `data/nuscenes`。

---

## 8. 建议的学习顺序

1. 打开 `samples/CAM_FRONT` 看几张图，再对照同时间戳的 `LIDAR_TOP`。
2. 读 `v1.0-mini/sample.json` 理解「一个 sample = 多传感器对齐的一帧」。
3. 跑通 `create_data.py`，看生成的 `nuscenes_infos_*.pkl`。
4. 再读 `docs/02_BEVFusion_算法结构说明.md`，把「数据字段」和「网络输入」对上。
