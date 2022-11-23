# coding:utf-8
import os
from PIL import Image


# bmp 转换为jpg
def bmpToJpg(file_path,src_dir):
    for fileName in os.listdir(file_path):
        print('--fileName--', fileName)  # 看下当前文件夹内的文件名字
        # print(fileName)
        newFileName = fileName[0:fileName.find(".")] + ".png"  # 改后缀
        print('--newFileName--', newFileName)
        im = Image.open(file_path + "/" + fileName)
        im.save(src_dir + "/" + newFileName)  # 保存到当前文件夹内


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "del " + file_path + "/*." + imageFormat
    os.system(command)


def main():
    file_path = r"D:\nextvpu\yanye\yanye_data\yanye_color_B_C\yanye_class_C02_C03\C03"
    src_dir = r'D:\nextvpu\yanye\yanye_data\yanye_color_B_C\yanye_class_C02_C03\C03_png'
    bmpToJpg(file_path,src_dir)
    # deleteImages(file_path, "bmp")


if __name__ == '__main__':
    main()
