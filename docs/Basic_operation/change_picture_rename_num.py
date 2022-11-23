# -*- coding:utf-8 -*-

import os
import random


class ImageRename():
    def __init__(self):
        self.path = r'D:\nextvpu\yanye\yanye_data\yanye_color_B_C\yanye_class_C02_C03\C03\\'
        print(self.path)

    def rename(self):
        filelist = os.listdir(self.path)
        random.shuffle(filelist)
        print(filelist)
        total_num = len(filelist)
        # new_filepath='/home/tanbin/deeplearning/python_learning/cainiaolearning/crawl_picture/tmp/'
        # if not os.path.exists(new_filepath):
        #     os.makedirs(new_filepath)

        i = 1  # 图片开始名称

        for item in filelist:
            # print item
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(i), '0>5s') + '.jpg')
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
