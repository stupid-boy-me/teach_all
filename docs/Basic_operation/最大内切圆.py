import cv2
import numpy as np


# 读取图片，转灰度
mask = cv2.imread("D:\\nextvpu\\yanye\\analyse_yansi\\C++\\2_acquire_max_square\\acquire_max_square\\144920.jpg")
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# 二值化
ret, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

# 识别轮廓
contours,hierarchy  = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 计算到轮廓的距离
raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
for i in range(mask_gray.shape[0]):
    for j in range(mask_gray.shape[1]):
        raw_dist[i, j] = cv2.pointPolygonTest(contours[0], (j, i), True)

# 获取最大值即内接圆半径，中心点坐标
minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)


# 半径：maxVal  圆心：maxDistPt
# 转换格式
maxVal = abs(maxVal)
radius = int(maxVal)

# 原图转换格式
result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)


# 绘制内切圆
cv2.circle(result, maxDistPt, radius, (0, 255, 0), 2, 1, 0)
# 绘制圆心
cv2.circle(result, maxDistPt, 1, (0, 255, 0), 2, 1, 0)
cv2.imshow('0', result)
cv2.waitKey(0)
