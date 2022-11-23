from tqdm import tqdm
import numpy as np
from PIL import Image
import os

input_dir = r"D:\nextvpu\yanye\yanye_data\data_norm_output_qiege\F03\\"
filenames = os.listdir(input_dir)
for filename in tqdm(filenames):
    image_path = os.path.join(input_dir, filename)

    img_L = np.array(Image.open(image_path).convert("L"))
    img_RGB = np.array(Image.open(image_path).convert("RGB"))

    non_color_0_0_0 = np.where(img_L != 0)[0].shape[0]

    pixel_sum = img_L.shape[0] * img_L.shape[1]
    if non_color_0_0_0/pixel_sum <= 0.95:
        os.remove(image_path)
    else:
        print("符合条件的非黑色像素个数：{} 占比：%{}".format(non_color_0_0_0,non_color_0_0_0/pixel_sum*100))

