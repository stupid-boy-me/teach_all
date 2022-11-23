from PIL import Image
import os
from tqdm import tqdm
def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('L', target_size, (0))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image

if __name__ == '__main__':
    in_path = r"D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\SegmentationClass_gray\\"
    out_path = r"D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\train_yanye_xia_B01\SegmentationClass_512_512\\"
    for name in tqdm(os.listdir(in_path)):
        image = Image.open(in_path + "/" + name).convert('L')#convert('RGB')
        size = (512, 512)
        new_image = pad_image(image, size)
        new_image.save(out_path + "/" + name)
