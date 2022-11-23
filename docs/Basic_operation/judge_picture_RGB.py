import os
import cv2
from PIL import Image
from tqdm import tqdm
input_dir = r"D:\nextvpu\yanye\yanye_data\yanye_ new_from_zhouyang\F03\\"
filenames = os.listdir(input_dir)
for filename in tqdm(filenames):
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path)
    if image.mode is not "RGB":
        print("{} is not RGB".format(image_path))
        image.close()
        os.remove(image_path)
    else:
        pass
