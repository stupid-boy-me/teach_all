import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

from utils.random_data import get_random_data, get_random_data_with_MixUp
from utils.utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始标签所在的路径
#   Out_VOCdevkit_path      输出标签所在的路径
#                                   处理后的标签为灰度图，如果设置的值太小会看不见具体情况。
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = "VOCdevkit_Origin"
Out_VOCdevkit_path      = "VOCdevkit"
#-----------------------------------------------------------------------------------#
#   Out_Num                 利用mixup生成多少组图片
#   input_shape             生成的图片大小
#-----------------------------------------------------------------------------------#
Out_Num                 = 5
input_shape             = [640, 640]

#-----------------------------------------------------------------------------------#
#   下面定义了xml里面的组成模块，无需改动。
#-----------------------------------------------------------------------------------#
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
    
tailstr = '''\
</annotation>
'''
if __name__ == "__main__":
    Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")
    
    Out_JPEGImages_path  = os.path.join(Out_VOCdevkit_path, "VOC2007/JPEGImages")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "VOC2007/Annotations")
    
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)
    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    xml_names = os.listdir(Origin_Annotations_path)

    def write_xml(anno_path, jpg_pth, head, input_shape, boxes, unique_labels, tail):
        f = open(anno_path, "w")
        f.write(head%(jpg_pth, input_shape[0], input_shape[1], 3))
        for i, box in enumerate(boxes):
            f.write(objstr%(str(unique_labels[int(box[4])]), box[0], box[1], box[2], box[3]))
        f.write(tail)
    
    #------------------------------#
    #   循环生成xml和jpg
    #------------------------------#
    for index in range(Out_Num):
        #------------------------------#
        #   获取两个图像与标签
        #------------------------------#
        sample_xmls = sample(xml_names, 2)
        unique_labels = get_classes(sample_xmls, Origin_Annotations_path)

        jpg_name_1  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[0])[0] + '.jpg')
        jpg_name_2  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[1])[0] + '.jpg')
        xml_name_1  = os.path.join(Origin_Annotations_path, sample_xmls[0])
        xml_name_2  = os.path.join(Origin_Annotations_path, sample_xmls[1])
            
        line_1 = convert_annotation(jpg_name_1, xml_name_1, unique_labels)
        line_2 = convert_annotation(jpg_name_2, xml_name_2, unique_labels)
        
        #------------------------------#
        #   各自数据增强
        #------------------------------#
        image_1, box_1  = get_random_data(line_1, input_shape) 
        image_2, box_2  = get_random_data(line_2, input_shape) 
        
        #------------------------------#
        #   合并mixup
        #------------------------------#
        image_data, box_data = get_random_data_with_MixUp(image_1, box_1, image_2, box_2)
        
        img = Image.fromarray(image_data.astype(np.uint8))
        img.save(os.path.join(Out_JPEGImages_path, str(index) + '.jpg'))
        write_xml(os.path.join(Out_Annotations_path, str(index) + '.xml'), os.path.join(Out_JPEGImages_path, str(index) + '.jpg'), \
                    headstr, input_shape, box_data, unique_labels, tailstr)
