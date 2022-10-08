import os
import random
from shutil import rmtree,copy
'''
将"D:\yiguohuang\study\pilipala_w\hyg\resnet\data\"路径下的多类数据变成格式如下的样子:
new_data
    -train
        -B01
        -B03
        -C02
    -val
        -B01
        -B03
        -C02 
'''
def makefile(filepath):
    if os.path.exists(filepath):
        rmtree(filepath)
    os.makedirs(filepath)

def main(root_path,split_rate=0.8):
    '''
    :param root_path:是项目的地址
    :return:
    '''
    data_path = os.path.join(root_path,"data")
    yanye_class = [cla for cla in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, cla))]  # ['B01', 'B03', 'C02']

    train_root = os.path.join(root_path,"data","train")
    makefile(train_root)

    # 建立训练集的文件夹
    for cla in yanye_class:
        makefile(os.path.join(train_root,cla))

    # 建立验证集的文件夹
    val_root = os.path.join(root_path,"data","val")
    makefile(val_root)
    for cla in yanye_class:
        makefile(os.path.join(val_root,cla))

    # 放图片的操作
    for cla in yanye_class: # ['B01', 'B03', 'C02']
        cla_path = os.path.join(data_path,cla) # D:\yiguohuang\study\pilipala_w\hyg\resnet\data\C02
        images = os.listdir(cla_path)
        num = len(images)
        train_picture = random.sample(images,k=int(num*split_rate))
        for index,image in enumerate(images):
            if image in train_picture:
                image_path = os.path.join(cla_path,image)
                new_path = os.path.join(train_root,cla)
                copy(image_path,new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            print("\r {} processing [{}/{}]".format(cla,index+1,num),end="\r")
        print()
    print("数据划分完毕")



if __name__ == '__main__':
    main(r"D:\yiguohuang\study\pilipala_w\hyg\resnet")

