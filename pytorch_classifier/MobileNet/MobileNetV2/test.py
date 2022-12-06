import os
import json
import sys
import logging
import torch
from PIL import Image
from torchvision import transforms
from MobileNetv2 import MobileNet_v2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default='/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals_output/test//',type=str,help='type of dataset')
parser.add_argument("--weights_path",default='/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv2_update/weights_folder/model-500-of-500-0.47060441970825195-0.6507537688442211.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default='/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv2_update/class_indices.json',type=str,help='json_path')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trf():
    data_transform = transforms.Compose(
        [
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
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
    for class_ in os.listdir(args.test_dir):
        # 每一个类别的路径
        class_path = os.path.join(args.test_dir,class_)
        all_num = len(os.listdir(class_path))
        for filename in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path,filename)
            if not os.path.exists(image_path):
                logging.info("image_path is not exists!!!")
            else:
                image = Image.open(image_path)
                image = trf()(image)
                image = torch.unsqueeze(image, dim=0)
                # 1.model 第一次放入GPU
                create_model = MobileNet_v2(theta=1,num_classes=10).to(device)
                # 2. model权重文件
                assert os.path.exists(args.weights_path), "file: '{}' dose not exist.".format(args.weights_path)
                # 3.权重文件放入模型中
                create_model.load_state_dict(torch.load(args.weights_path, map_location=device))
                create_model.eval()
                with torch.no_grad():  # 第二次   数据需要放入GPU
                    output = torch.squeeze(create_model(image.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy() # # predict_cla 3
                    predict_class = class_indict[str(predict_cla)] # 预测的类别,要跟实际的类别进行一个比较
                    if predict_class == class_:
                        correct_num += 1

        logging.info("类别{}的准确率是{}".format(class_, correct_num / all_num))
        correct_num = 0
    logging.info("================>>结束预测！<<================")


if __name__ == '__main__':
    main()
