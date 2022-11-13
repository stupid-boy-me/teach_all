# 工具类
import os
import random
from shutil import copy2
from tqdm import tqdm
import json

'''既可以数据划分，也可以直接生成list'''
def data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.2, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\resnet\data_animals_ten\animals10\raw-img
    :param target_data_folder: 目标文件夹 E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\resnet\data_animals_ten\animals10\data_output
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    print("class_names",class_names)
    '''第一步：在目标目录下创建文件夹'''
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    '''第二步：按照比例划分数据集，并进行数据图片的复制'''
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息
    # 首先进行分类遍历
    class_indices = dict((k, v) for v, k in enumerate(class_names))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=9)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    for class_name in tqdm(class_names):
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)  # 每一个类别的总数量 132
        current_data_index_list = list(range(current_data_length))  # [0,1,2,...,131]
        random.shuffle(current_data_index_list)  # 随机打乱
        image_class = class_indices[class_name]
        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        '''添加内容'''

        for i in current_data_index_list:  # 打乱后的[0,1,2,...,131]
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
                '''添加内容'''
                train_images_path.append(src_img_path)
                train_images_label.append(image_class)
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
                '''添加内容'''
                val_images_path.append(src_img_path)
                val_images_label.append(image_class)
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1
                '''添加内容'''
                test_images_path.append(src_img_path)
                test_images_label.append(image_class)
            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))
        # print("train_images_path",train_images_path)
        # print("train_images_label",train_images_label)
        # print("val_images_path",val_images_path)
        # print("val_images_label",val_images_label)
    return train_images_path, train_images_label, val_images_path, val_images_label

if __name__ == '__main__':
    src_data_folder = "/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals"
    target_data_folder = "/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals_output"
    data_set_split(src_data_folder, target_data_folder)
