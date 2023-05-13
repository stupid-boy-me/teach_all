'''
input：xml
output：txt
    img_name x1 y1 x2 y2 label,[img_name x1 y1 x2 y2 label]
    2007_000032.jpg 104 78 375 183 0 133 88 197 123 0 195 180 213 229 14 26 189 44 238 14
'''

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects

if __name__ == '__main__':

    txt_file = open(r'Test_delete\voc2012toyolo.txt','w')  # 输出txt的格式命名文件
    Annotations = r'\VOC\dataset\VOCdevkit\VOC2012\Annotations\\'  # VOC的Annotations路径

    xml_files = os.listdir(Annotations)
    count = 0
    for xml_file in xml_files:
        count += 1
        image_path = xml_file.split('.')[0] + '.jpg'
        results = parse_rec(Annotations + xml_file)
        print(results)
        if len(results)==0:
            print(xml_file)
            continue
        txt_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
        txt_file.write('\n')
    txt_file.close()
