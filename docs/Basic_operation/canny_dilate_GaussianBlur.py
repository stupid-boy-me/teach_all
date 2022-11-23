import os

import cv2
from tqdm import tqdm

def canny_dilate_GaussianBlur(path):
    img = cv2.imread(path)
    data = (150, 150)
    img_copy = img.copy()
    imgCanny = cv2.Canny(img, *data)
    # 创建矩形结构
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # 膨化处理
    # 更细腻
    img_dilate = cv2.dilate(imgCanny, g)
    # 更粗大
    img_dilate2 = cv2.dilate(imgCanny, g2)

    shape = img_dilate.shape
    # 提取
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate2[i, j] == 0:  # 二维定位到三维
                img[i, j] = [0, 0, 0]

    dst = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate[i, j] != 0:  # 二维定位到三维
                img_copy[i, j] = dst[i, j]

    return img_copy

if __name__ == '__main__':
    path = r"D:\nextvpu\OCR\book_object\data\cornel_data\task1\picture_book_background"
    filenames = [filename for filename in os.listdir(path)]
    for file in tqdm(filenames):
        file_path = os.path.join(path,file)
        output = canny_dilate_GaussianBlur(file_path)
        output_path = r"D:\nextvpu\OCR\book_object\data\cornel_data\task1\picture_canny_dilate_GaussianBlur"
        output_canny_dilate_GaussianBlur = os.path.join(output_path+ "/" , file.split('.')[0] + ".jpg")
        print(output_canny_dilate_GaussianBlur)
        # cv2.imwrite(output_canny_dilate_GaussianBlur, output)
