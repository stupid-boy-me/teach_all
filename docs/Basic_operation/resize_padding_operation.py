from PIL import Image
import os

def resize_padding(picture_path):
    n = 0
    for filename in os.listdir(picture_path):
        filename_path = os.path.join(picture_path,filename)
        target_size = [512, 512]
        img1 = Image.open(filename_path)

        iw, ih = img1.size  # 原始图像的尺寸
        w, h = target_size  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        new_img = img1.resize((nw, nh), Image.BILINEAR)  # 缩小图像

        img = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像


        # // 为整数除法，计算图像的位置
        img.paste(new_img, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式
        img.save(r"D:\nextvpu\yanye\svm_classifier\\20220525data\C02_512/{}.jpg".format(n))
        n += 1

if __name__ == '__main__':
    path = r'D:\nextvpu\yanye\svm_classifier\20220525data\C02_C03\\'
    resize_padding(path)
