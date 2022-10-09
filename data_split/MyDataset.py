import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


"""自定义数据集"""
class MyDataSet(Dataset):
    '''
     读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
     :param images_path: 图片的路径，是一个list
     :param images_class: 对应图片的标签，也是一个list
     :param transform:  数据增广

     :return:
     '''
    def __init__(self,images_path,images_class,transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __getitem__(self, index):
        # 通过index对images_path 进行读取
        # 输入 index
        # 输出
        image = Image.open(self.images_path[index])
        if image is None:
            print("图像为空")
        label = self.images_class[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels