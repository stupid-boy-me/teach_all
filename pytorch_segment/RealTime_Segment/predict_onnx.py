import argparse
import os
import torch

import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms
import onnxruntime as ort
# from scipy.signal import find_peaks
from skimage.draw import line




city_mean, city_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])



def sample_lane_by_h_samples(lane_lines, h_samples, img_width=336):
    h_samples = np.array(h_samples, dtype=np.float64)
    
    for lane in lane_lines:
        x1, y1 = lane['start_point']  # (x, y)
        x2, y2 = lane['end_point']
        
        # 转为 float 防止整数除法
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # 初始化 x 为 -2（无效）
        x_values = np.full_like(h_samples, -2.0)
        
        # 计算 y 范围
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        # 找出在 [y_min, y_max] 内的采样行
        valid_mask = (h_samples >= y_min) & (h_samples <= y_max)
        
        if np.any(valid_mask):
            y_targets = h_samples[valid_mask]
            
            # 处理水平线（y1 == y2）
            if abs(y2 - y1) < 1e-6:
                # 水平线：所有有效 y 对应同一个 x（取中点或 x1）
                x_interp = (x1 + x2) / 2.0
                x_values[valid_mask] = x_interp
            else:
                # 标准线性插值：x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                x_interp = x1 + (x2 - x1) * (y_targets - y1) / (y2 - y1)
                x_values[valid_mask] = x_interp
        
        # 裁剪到图像宽度
        x_values = np.clip(x_values, 0, img_width - 1)
        x_values = np.round(x_values).astype(np.int32)
        
        # 构建 (x, y) 点
        sampled_points = [(int(x), int(y)) for x, y in zip(x_values, h_samples.astype(int))]
        lane['sampled_points'] = sampled_points
    
    return lane_lines


def PostProcess(instance_pred, input_feature_map_ratio, save_rows=True):
    """
    Args:
        instance_pred: numpy array (H, W, C), 嵌入向量
        save_rows: 保留两条车道线的参数
        input_feature_map_ratio : 输入和输出特征图的倍率
    Returns:
        instance_mask: numpy array (H, W), 实例分割掩码 (0=背景, 1~4=不同车道线)
        lane_lines: list of dict, 每条线包含:
            - 'start_point': (x, y) 起点（图像底部）
            - 'end_point': (x, y) 终点（图像顶部）
            - 'label': 新分配的标签 (0,1,2,3)
            - 'points': 原始点集 (N,2) 格式为 (x, y)
    """
    # 🟢 转为 torch.Tensor 以便复用你原有的后处理（也可纯 numpy 改写）
    instance_pred = torch.from_numpy(instance_pred)
    x_center = instance_pred.shape[3] * input_feature_map_ratio / 2.0
    # 👇 以下和你原来代码完全一致 👇
    instance_map = torch.argmax(instance_pred, dim=1)  # shape: (1, H, W)
    instance_map = instance_map.squeeze(0).cpu().numpy()  # shape: (384, 672)

    
    labels = np.unique(instance_map)

    print(f"检测到的实例ID: {labels}")

    label_counts = {}
    lane_lines = []  # 存储每条线的信息
    for inst_id in labels:
        import random
        if inst_id == 0:  # 跳过背景
            continue
        mask = (instance_map == inst_id)
        if mask.sum() == 0:
            continue

        valid_coords = np.column_stack(np.where(mask))  # (row, col) = (v, u)

        # ADD
        valid_coords[:, 0] = valid_coords[:, 0] * input_feature_map_ratio  # x 坐标乘以常量
        valid_coords[:, 1] = valid_coords[:, 1] * input_feature_map_ratio  # y 坐标乘以常量

        label_counts[inst_id] = valid_coords
        if len(valid_coords) < 2:
            continue

        # 提取 x, y
        ys = valid_coords[:, 0]  # 行坐标（高度方向）
        xs = valid_coords[:, 1]  # 列坐标（宽度方向）
        # 多项式拟合（一次曲线 = 直线）
        try:
            coeffs = np.polyfit(ys, xs, deg=1)  # 拟合 x = f(y)
            poly = np.poly1d(coeffs)
        except:
            continue

        # 定义采样范围：从图像底部到顶部
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        y_sample = np.linspace(y_min, y_max, 50)
        x_sample = poly(y_sample)

        # 过滤超出图像边界的点  [图像的边界也需要进行扩大, 上一个点进行缩放到672 *384了 图像的边界也要进行扩大]
        valid = (x_sample >= 0) & (x_sample < instance_pred.shape[3] * input_feature_map_ratio) & (y_sample >= 0) & (y_sample < instance_pred.shape[2] * input_feature_map_ratio)
        if not np.any(valid):
            continue
        x_sample = x_sample[valid]
        y_sample = y_sample[valid]

        if len(x_sample) < 2:
            continue

        # 起点 = 最底部点 (y最大), 终点 = 最顶部点 (y最小)
        start_point = (int(x_sample[-1]), int(y_sample[-1]))   # 底部
        end_point = (int(x_sample[0]), int(y_sample[0]))       # 顶部

        # 计算中点 x 坐标（用于左右分组）
        mid_x = (start_point[0] + end_point[0]) / 2.0
        mid_y = (start_point[1] + end_point[1]) / 2
        lane_lines.append({
            'start_point': start_point,   # (x, y)
            'end_point': end_point,       # (x, y)
            'mid_x': mid_x,               # 中点 x，用于分组
            'mid_y': mid_y,        # 最小 y（顶部点 y），用于排序
            'points': valid_coords,
        })


    # ======== 新增：按中点划分左右，再按 min_y 排序分配标签 ========

    left_lines = []
    right_lines = []

    for line in lane_lines:
        if line['mid_x'] < x_center:
            left_lines.append(line)
        else:
            right_lines.append(line)

    # 左侧：按 min_y 升序排序（y 越小越靠上）
    left_lines.sort(key=lambda x: x['mid_y'], reverse = True)
    # 右侧：按 min_y 升序排序
    right_lines.sort(key=lambda x: x['mid_y'], reverse = True)

    # 分配标签
    for i, line in enumerate(left_lines):
        line['label'] = 1 if i == 0 else 0  # 第一条是1，其余是0

    for i, line in enumerate(right_lines):
        line['label'] = 2 if i == 0 else 3  # 第一条是2，其余是3

    # 合并回 lane_lines（保持原始顺序或按标签排序）
    lane_lines = left_lines + right_lines

    if save_rows:
        # 只保留标签是1和2的
        lane_lines = [lane for lane in lane_lines if lane['label'] in (1, 2)]
    
    tu_simple_h_samples = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

    # 原图高度 720 → 你的图高度 384
    scale = instance_map.shape[0] * input_feature_map_ratio / 720.0
    samples = [int(round(h * scale )) for h in tu_simple_h_samples]

    # 采样
    result = sample_lane_by_h_samples(
        lane_lines=lane_lines,
        h_samples=samples,
        img_width=instance_map.shape[1] * input_feature_map_ratio
    )
    return result

# ========== 替换 load_model ==========

def load_onnx_model(onnx_path):
    # 创建推理会话
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return session



def preprocess_cv2_image(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0  # HWC -> CHW, [0,1]

    # city_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    # city_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    # image = (image - city_mean) / city_std

    image = (image.astype(np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (
        0.229,
        0.224,
        0.225,
    )

    image = image.transpose(2, 0, 1).astype(np.float32) # [np.newaxis, ...]
    return image



def draw_vanishing_point_analysis(bgr_image, mask, start_ratio=0.5, 
                                  point_color=(0, 255, 255),      # 黄色：边界点
                                  left_line_color=(255, 0, 0),    # 蓝色：左线
                                  right_line_color=(0, 0, 255),   # 红色：右线
                                  vp_color=(0, 255, 0),           # 绿色：消失点
                                  line_thickness=2,
                                  point_radius=3):
    """
    在原图上绘制：
    - 左右边界点（下半部分）
    - 拟合的左右边界延长线
    - 两条线的交点（消失点）

    Args:
        bgr_image: 原图，BGR 格式，np.ndarray, shape (H, W, 3)
        mask: 二值 mask，值为 0 或 1，shape (H, W)
        start_ratio: 从图像高度的 start_ratio 开始扫描（如 0.5）
    
    Returns:
        result_image: 带有可视化元素的图像
    """
    H, W = mask.shape
    result = bgr_image.copy()
    
    # Step 1: 找边界点
    start_row = int(H * start_ratio)
    left_points = []
    right_points = []

    for y in range(start_row, H):
        row = mask[y]
        if row.sum() == 0:
            break
        xs = np.where(row == 1)[0]
        left_x = xs[0]
        right_x = xs[-1]
        if abs(right_x - left_x) >= 50:
            left_points.append((int(left_x + 5), int(y)))
            right_points.append((int(right_x - 5), int(y)))
            # 可选：绘制边界点
            cv2.circle(result, (left_x, y), point_radius, point_color, -1)
            cv2.circle(result, (right_x, y), point_radius, point_color, -1)

    if len(left_points) < 2 or len(right_points) < 2:
        print("Not enough points to fit lines.")
        return result

    # Step 2: 拟合直线
    left_pts = np.array(left_points, dtype=np.float32)
    right_pts = np.array(right_points, dtype=np.float32)

    left_line = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()  # [vx, vy, x0, y0]
    right_line = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # # Step 3: 计算交点
    def line_intersection(line1, line2):
        vx1, vy1, x1, y1 = line1
        vx2, vy2, x2, y2 = line2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t1, _ = np.linalg.solve(A, b)
        x = x1 + t1 * vx1
        y = y1 + t1 * vy1
        return (float(x), float(y))

    vp = line_intersection(left_line, right_line)

    # Step 4: 绘制延长线（覆盖整张图像高度范围）
    def draw_line(img, line, color, thickness):
        vx, vy, x0, y0 = line
        # 生成两个端点：让线穿过整个图像（上下边界）
        if abs(vx) < 1e-6:  # 垂直线
            pt1 = (int(x0), 0)
            pt2 = (int(x0), H)
        else:
            # y = y0 + (vy/vx)(x - x0)
            # 当 y=0:
            x_top = x0 - y0 * vx / vy if abs(vy) > 1e-6 else x0
            # 当 y=H-1:
            x_bottom = x0 + (H - 1 - y0) * vx / vy if abs(vy) > 1e-6 else x0
            pt1 = (int(x_top), 0)
            pt2 = (int(x_bottom), H - 1)
        cv2.line(img, pt1, pt2, color, thickness)
        return pt1, pt2

    draw_line(result, left_line, left_line_color, line_thickness)
    draw_line(result, right_line, right_line_color, line_thickness)

    # Step 5: 绘制消失点（即使在图像外，也标在最近边界或画十字）
    if vp is not None:
        x, y = vp
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(result, (int(x), int(y)), 8, vp_color, -1)
            cv2.circle(result, (int(x), int(y)), 10, (0, 0, 0), 2)
        else:
            # 在图像边界上画一个十字表示方向
            cv2.drawMarker(result, (int(x), int(y)), vp_color, markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)

    # 可选：绘制检测到的边界点（取消注释下面两行）
    # for pt in left_points:
    #     cv2.circle(result, pt, point_radius, point_color, -1)
    # for pt in right_points:
    #     cv2.circle(result, pt, point_radius, point_color, -1)

    return result, vp



def get_boundary_points(H, W, step=2):
    """
    按顺序生成边界点列表（支持步长控制）：
    1. 左边界下半段: (x=0, y=H//2 → H-1)      [从上到下]
    2. 底边:         (y=H-1, x=0 → W-1)       [从左到右]
    3. 右边界下半段: (x=W-1, y=H-1 → H//2)    [从下到上]

    Args:
        H, W: 图像高宽
        step: 采样步长（正整数），默认为1（全采样）

    Returns:
        points: list of (row, col) = (y, x)
    """
    if step < 1:
        raise ValueError("step must be >= 1")

    points = []

    # 1. 左边界下半段: y from H//2 to H-1
    for y in range(H // 2, H, step):
        points.append((y, 0))

    # 2. 底边: x from 0 to W-1
    y_bottom = H - 1
    for x in range(0, W, step):
        points.append((y_bottom, x))

    # 3. 右边界下半段: y from H-1 down to H//2
    x_right = W - 1
    # 使用 range(start, stop, step)，start=H-1, stop=H//2-1, step=-step
    y = H - 1
    while y >= H // 2:
        points.append((y, x_right))
        y -= step

    return points

def count_ones_and_get_coordinates(y_vp_int, x_vp_int, y_p, x_p, mask):
    H, W = mask.shape
    # 获取 vp 到 p 的所有像素坐标
    rr, cc = line(y_vp_int, x_vp_int, y_p, x_p)  # rr=y, cc=x

    # 过滤掉超出图像边界的点（理论上不会，但安全起见）
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr, cc = rr[valid], cc[valid]

    # 统计 mask[rr, cc] 中 1 的个数
    num_ones = np.sum(mask[rr, cc] == 1)
    
    # 获取 mask[rr, cc] 中 1 的坐标
    ones_coordinates = np.column_stack((rr[mask[rr, cc] == 1], cc[mask[rr, cc] == 1]))

    return num_ones, ones_coordinates


def count_ones_along_lines_from_vp(mask, vp):
    """
    对每个边界点 p，计算线段 vp -> p 上 mask 中值为 1 的像素个数。

    Args:
        mask: 2D np.ndarray, shape (H, W), values in {0, 1}
        vp: tuple (x, y) —— 注意：是 (x, y)，即 (col, row)

    Returns:
        counts: list of int, 长度 = 边界点数量
    """
    H, W = mask.shape
    x_vp, y_vp = vp  # vp 是 (x, y)
    
    # 转为整数坐标（line 要求整数）
    x_vp_int = int(round(x_vp))
    y_vp_int = int(round(y_vp))

    boundary_points = get_boundary_points(H, W, 2)  # list of (row, col) = (y, x)
    counts = []
    coordinates = []
    for (y_p, x_p) in boundary_points:
        num_ones, ones_coordinates = count_ones_and_get_coordinates(y_vp_int, x_vp_int, y_p, x_p, mask)
        
        counts.append(int(num_ones))
        coordinates.append(ones_coordinates)
    return counts, coordinates, boundary_points


def find_all_significant_peaks(values, 
                               window_size=10,
                               rel_height_threshold=0.3,
                               min_peak_distance=20):
    """
    找所有显著峰值的原始索引。
    
    Args:
        values: list or 1D array
        window_size: 用于估计局部背景（可选，当前用于动态阈值）
        rel_height_threshold: 峰值需 >= global_max * rel_height_threshold
        min_peak_distance: 峰之间最小距离（防密集检测）
    
    Returns:
        peak_indices: list of int
    """
    values = np.array(values, dtype=np.float32)
    n = len(values)
    if n < 3:
        return []

    global_max = values.max()
    if global_max == 0:
        return []

    min_height = rel_height_threshold * global_max

    # Step 1: 找所有局部最大值（严格：左<中>=右）
    local_max_indices = []
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            if values[i] >= min_height:
                local_max_indices.append(i)

    # 如果没有，尝试放宽条件（>= 两边）
    if not local_max_indices:
        for i in range(1, n - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                if values[i] >= min_height:
                    local_max_indices.append(i)

    if not local_max_indices:
        return []

    # Step 2: 按高度降序排序，用于距离过滤（贪心保留高的）
    local_max_indices = np.array(local_max_indices)
    heights = values[local_max_indices]
    sorted_idx = np.argsort(-heights)  # 从高到低
    sorted_peaks = local_max_indices[sorted_idx]

    # Step 3: 距离过滤（保留高且不近的）
    final_peaks = []
    for peak in sorted_peaks:
        # 检查是否与已选峰太近
        too_close = False
        for p in final_peaks:
            if abs(peak - p) < min_peak_distance:
                too_close = True
                break
        if not too_close:
            final_peaks.append(peak)

    # 按原始顺序返回
    final_peaks.sort()
    return final_peaks

def select_peaks_with_side(final_peaks, coordinates,index):
    """
    返回始终包含 'left' 和 'right' 键的字典。
    无对应峰值时，值设为 -1。
    """
    # 初始化结果
    result = {"left": -1, "right": -1}

    if not final_peaks:
        return result

    final_peaks = sorted(final_peaks)
    n = len(final_peaks)

    if n == 1:
        p = final_peaks[0]
        if p <= index:
            result["left"] = coordinates[p]
        else:
            result["right"] = coordinates[p]

    elif n == 2:
        p1, p2 = final_peaks
        result["left"] = coordinates[p1]
        result["right"] = coordinates[p2]

    else:  # n >= 3
        left_candidates = [p for p in final_peaks if p < index]
        right_candidates = [p for p in final_peaks if p > index]

        if left_candidates:
            result["left"] = coordinates[max(left_candidates)][::10]  # 最靠近 index 的左侧

        if right_candidates:
            result["right"] = coordinates[min(right_candidates)][::10]   # 最靠近 index 的右侧

    return result


# 图像测试
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')

    parser.add_argument('--model_weight', type=str, default='/algdata01/yiguo.huang/project_code/NextVpu/UFLDv2/liuxiao/FastSCNNSimpleInstanceSegENetCrossConvTestunsampleAugment_672_384_FDMobileNet/cpt_FDMobilenet_diceloss/FastSCNN_FDMobilenet_backbone_384_672_model-sim.onnx', help='Pretrained model weight')
    parser.add_argument('--input_pic', type=str, default='/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SegmentLine/Make_CVAT_数据堂/20250201_002016_main_cvat1018/20250201_002016_main_cvat1018/images/20250201_002016_main_frame_07260.jpg', help='Path to the input picture')
    # /algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SegmentLine/Make_CVAT_数据堂/20250201_002016_main_cvat1018/20250201_002016_main_cvat1018/images/20250201_002016_main_frame_07260.jpg
    # args parse
    args = parser.parse_args()
    model_weight, input_pic = args.model_weight, args.input_pic
    # path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SCNNMerge/leftImg8bit/val"
    # path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SCNN/20250201_002016_main_cvat1018/images/val/"
    
    # path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SCNN/20250328sel_imgs1200_1080p_cvat1014/images/train/"
    # one
    # path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SCNN/20250415sel_imgs2120_1080p_cvat1015/images/train/"
    # cvat_basename = '20250415sel_imgs2120_1080p_cvat1015_0.5_augment'
    # two
    path_root = "/algdata01/yiguo.huang/project_code/NextVpu/UFLDv2/liuxiao/FastSCNNSimpleInstanceSegENetCrossConvTestunsampleAugment_672_384_FDMobileNet/test_images/20250201_001020_main-backward"
    path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SegmentLine/Make_CVAT_数据堂/20250201_002016_main_cvat1018/20250201_002016_main_cvat1018/images/"
    path_root = "/algdata03/common/autodrive/saferider/lane2d/SegPoseLine/SegmentLine/Make_CVAT_数据堂/task_20251015_railway_2025_10_21_09_40_15_cvat1216/task_20251015_railway_2025_10_21_09_40_15_cvat1216/images/"
    cvat_basename = '20250201_001020_main-backward'
    args.save = os.path.join("/algdata01/yiguo.huang/project_code/NextVpu/UFLDv2/liuxiao/FastSCNNSimpleInstanceSegENetCrossConvTestunsampleAugment_672_384_FDMobileNet/vis_seg", cvat_basename)
    
    os.makedirs(args.save, exist_ok=True)
    filenames = [filename for filename in os.listdir(path_root) if "jpg" in filename]


    session = load_onnx_model(model_weight)

    for filename in tqdm(filenames[:10]):
        input_pic = os.path.join(path_root, filename)
        image = cv2.imread(input_pic)
        H_cv, W_cv = image.shape[:2]


        basename = os.path.basename(input_pic)
        # cv2.imwrite("/algdata01/yiguo.huang/project_code/NextVpu/UFLDv2/liuxiao/FastSCNNSimpleInstanceSegENetCrossConv18ms_nearestFull/vis_seg/{}_ori.jpg".format(basename),image_cv)
        # 在672 * 384 上进行可视化
        filename1 = basename.split(".")[0]

        image_cv_resize = cv2.resize(image, (672, 384))  # (W, H)  # 这个不需要动 前处理的输入

        # ✅ 预处理：cv2 → numpy CHW float32
        input_tensor = preprocess_cv2_image(image_cv_resize)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, 384, 672)

        # ✅ ONNX 推理
        instance_pred = session.run(
            ['instance_pred'],  # 输出名需与导出时一致
            {'input': input_tensor}
        )[0]


        instance_pred = torch.from_numpy(instance_pred)

        # 👇 以下和你原来代码完全一致 👇
        instance_map = torch.argmax(instance_pred, dim=1)  # shape: (1, H, W)
        instance_map = instance_map.squeeze(0).cpu().numpy()  # shape: (384, 672)
     
        # 找到两侧
        # 假设你有一个 mask，shape=(480, 640)

        # image_cv_resize_vis, vp = draw_vanishing_point_analysis(image_cv_resize, instance_map, start_ratio=0.6)

        # values, coordinates, boundary_points = count_ones_along_lines_from_vp(instance_map, vp)

        # final_peaks = find_all_significant_peaks(values, 15)

        # mid_points = (int(instance_map.shape[1] / 2), 0)
        # index = boundary_points.index(mid_points)
        # final_lanes_points = select_peaks_with_side(final_peaks, coordinates, index)

        # '''
        pred = instance_map
        pred_255 = pred * 255
            
        # 将单通道图像转换为BGR格式
        pred_255_bgr = cv2.cvtColor(pred_255.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # 将mask图转换为颜色映射，以便更好地可视化
        colormap = cv2.COLORMAP_JET
        mask_colored = cv2.applyColorMap(pred_255_bgr, colormap)
       
        # 将单通道BGR图像与原始BGR图像合并
        merged_image = cv2.addWeighted(image_cv_resize, 0.5, mask_colored, beta = 0.5 , gamma = 0 )


        cv2.imwrite(os.path.join(args.save, basename),merged_image)
        # cv2.imwrite(os.path.join(args.save, filename1 + "_vis.jpg"),image_cv_resize)
        # '''
        
        # for key, values in final_lanes_points.items():
        #     import random
        #     color_vis = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        #     for value in values:
        #         cv2.circle(image_cv_resize, (int(value[1]), int(value[0])), radius = 2, color=color_vis, thickness = 2)
            
       
        # cv2.imwrite(os.path.join(args.save, basename.split('.')[0] + "final_.jpg"),image_cv_resize)

        # assert False
       