import os
import random

'''
以口罩检测数据集为例
data格式
--VOCdevkit
    --MaskVoc(类似于VOC2007、VOC2012)
        --Annotations
        --ImageSets
            --Main
                --trainval.txt
                --test.txt
                --train.txt
                --val.txt
        -- JPEGImages
    
'''



trainval_percent = 0.9  # 训练和验证集所占比例，剩下的0.1就是测试集的比例
train_percent = 0.8  # 训练集所占比例，可自己进行调整
xmlfilepath = r'yolov1\data\Annotations'
txtsavepath = r'yolov1\VOCdevkit\MaskVoc\ImageSets\Main'
total_xml = os.listdir(xmlfilepath)
# print(total_xml)
num = len(total_xml)
list = range(num)
# print(list)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(r'yolov1\VOCdevkit\MaskVoc\ImageSets\Main/trainval.txt', 'w')
ftest = open(r'yolov1\VOCdevkit\MaskVoc\ImageSets\Main/test.txt', 'w')
ftrain = open(r'yolov1\VOCdevkit\MaskVoc\ImageSets\Main/train.txt', 'w')
fval = open(r'yolov1\VOCdevkit\MaskVoc\ImageSets\Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    # print(name)
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

