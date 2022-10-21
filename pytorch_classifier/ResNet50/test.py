import os
import json
import sys
import logging
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import ResNet
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default=r'E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\1_resnet\data_animals_ten\animals10\raw_img_output\test\cat',type=str,help='type of dataset')
parser.add_argument("--weights_path",default=r'E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\1_resnet\weights\model-999 of 1000-0.056281160563230515-0.8586387434554974.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default=r'E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\1_resnet\class_indices.json',type=str,help='json_path')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trf():
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transform



def main():
    correct_num = 0
    logging.info("================>>开始预测<<================")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.json_path, "r") as f:
        class_indict = json.load(f)
    # 加载图像
    filenames = os.listdir(args.test_dir)
    all_num = len(filenames)
    for filename in tqdm(filenames):
        image_path = os.path.join(args.test_dir,filename)
        if not os.path.exists(image_path):
            logging.info("image_path is not exists!!!")
        else:
            image = Image.open(image_path)
            image = trf()(image)
            image = torch.unsqueeze(image, dim=0)

            # 1.model 第一次放入GPU
            create_model = ResNet(num_class=10).to(device)
            # 2. model权重文件
            assert os.path.exists(args.weights_path), "file: '{}' dose not exist.".format(args.weights_path)
            # 3.权重文件放入模型中
            create_model.load_state_dict(torch.load(args.weights_path, map_location=device))

            create_model.eval()
            with torch.no_grad():  # 第二次   数据需要放入GPU
                output = torch.squeeze(create_model(image.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                # logging.info("{}路径图像的类别是:==>{}!".format(image_path,class_indict[str(predict_cla)]))
            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
            '''打印类别信息和所有类别的准确率'''
            plt.title(print_res)
            for i in range(len(predict)):
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                          predict[i].numpy()))
            '''统计一个类别的准确率'''
            if class_indict[str(predict_cla)] == image_path.split('\\')[-2]:
                correct_num += 1




    logging.info("类别{}的准确率是{}".format(class_indict[str(predict_cla)],correct_num/all_num))
    logging.info("================>>结束预测！<<================")


if __name__ == '__main__':
    main()