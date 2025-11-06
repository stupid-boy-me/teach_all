import os
import os.path as osp
import random
from tqdm import tqdm

import cv2
import numpy as np
from cityscapesscripts.helpers.labels import trainId2label
from torch.utils.data import Dataset
from torchvision import transforms as transforms_vision
from PIL import Image, ImageEnhance
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from collections import namedtuple
from utils import transforms


city_mean, city_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms_vision.Compose([transforms_vision.ToTensor(), transforms_vision.Normalize(city_mean, city_std)])

palette = []
for key in sorted(trainId2label.keys()):
    if key != -1 and key != 255:
        palette += list(trainId2label[key].color)


class Cityscapes(Dataset):
    """
       Cityscapes dataset is employed to load train or val set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        split: train, val
        crop_size: (512, 1024), only works for 'train' split
        mean: rgb_mean (0.485, 0.456, 0.406)
        std: rgb_mean (0.229, 0.224, 0.225)
        ignore_label: 255
    """

    # Codes are based on https://github.com/mcordts/cityscapesScripts
    
    # --------------------------------------------------------- #
    # a label and all meta information
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )

    #--------------------------------------------------------------------------------
    # A list of all labels
    #--------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!

    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]


    # 将原始标签图像（含0-33的ID值）转换为训练用标签（0-18的有效类别+255忽略类）
    '''
    这种映射方式是语义分割任务中处理Cityscapes数据集的标准做法，确保模型只关注有意义的类别，忽略无效区域。
    '''
    id_to_train_id = np.array([label.trainId for label in labels])
    def __init__(self, root, split='train', crop_size=(384, 672), mean=city_mean, std=city_std, ignore_label=255):
        
        img_dir = os.path.join(root, 'leftImg8bit', split)
        msk_dir = os.path.join(root, 'gtFine', split)


        self.split = split
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.files = []

        # Augmentation
        
        # 训练集数据增强参数（针对Cityscapes语义分割任务）
        self.scale = 1.0
        self.randscale = 0.25  # 随机缩放范围 [-25%, +25%]
        self.brightness = 0.2  # 亮度变化范围
        self.contrast = 0.2    # 对比度变化范围
        self.saturation = 0.2  # 饱和度变化范围
        self.h_flip = 0.5      # 水平翻转概率
        self.v_flip = 0.0      # 垂直翻转概率（道路场景通常不需要）

        if self.split == 'train':
            self.transform = AT.Compose([
                transforms.Scale(scale=self.scale),
                AT.RandomScale(scale_limit=self.randscale),
                AT.PadIfNeeded(min_height=self.crop_h, min_width=self.crop_w, value=(114,114,114), mask_value=(0,0,0)),
                AT.RandomCrop(height=self.crop_h, width=self.crop_w),
                AT.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                AT.HorizontalFlip(p=self.h_flip),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),                
            ])

        elif self.split == 'val':
            self.transform = AT.Compose([
                transforms.Scale(scale=self.scale),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        for city in os.listdir(img_dir):
            if city.endswith('.txt') == True or city.endswith('.py') == True: continue
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(msk_dir, city)

            for file_name in os.listdir(city_img_dir):
                img_file = os.path.join(city_img_dir, file_name)
                mask_name = f"{file_name.split('_leftImg8bit')[0]}_gtFine_labelIds.png"
                label_file = os.path.join(city_mask_dir, mask_name)
                self.files.append({'img': img_file, 'label': label_file})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        datafiles = self.files[index]
        image = np.array(Image.open(datafiles['img']).convert('RGB'))
        mask = np.array(Image.open(datafiles['label']).convert('L'))
        
        # Perform augmentation and normalization
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        # Encode mask using trainId
        mask = self.encode_target(mask)
        return image, mask
    

    @classmethod
    def encode_target(cls, mask):
        return cls.id_to_train_id[np.array(mask)]
    

if __name__ == '__main__':
    dataset = Cityscapes(root='/algdata02/yiguo.huang/Data/cityscapes/', split='train', crop_size=(512, 1024))
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)