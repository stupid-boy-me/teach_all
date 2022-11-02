import os
import shutil
from PIL import Image
img_path = r'E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\resnet\data_animals_ten\animals10\data_output\test\cat'
filenames = os.listdir(img_path)
for filename in filenames:
    image_path = os.path.join(img_path,filename)
    fp = open(image_path, 'rb')
    image = Image.open(fp)
    fp.close()
    if image.mode != 'RGB':
        os.remove(image_path)
        print('删除成功')

