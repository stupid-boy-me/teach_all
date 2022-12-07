import os
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import sys
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import logging
import torch
from torchvision import transforms
from Mobilenetv3 import MobileNetV3_large,MobileNetV3_small
import argparse
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
 
parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default='/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals_output/test/',type=str,help='type of dataset')
parser.add_argument("--weights_path",default='/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3_with_Confusion_Matrix/weights_folder/model-10-of-500-1.1974492073059082-0.5236180904522613.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default='/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3/class_indices.json',type=str,help='json_path')
parser.add_argument('--num_classes', default=5, type=int,help='class_num for training')
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

 
def trf():
    data_transform = transforms.Compose(
        [
        transforms.RandomResizedCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transform


def test(model, test_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # 加载测试集和预训练模型参数
    class_list = list(os.listdir(os.path.join(args.test_dir)))
    class_list.sort()
    print(class_list)

    test_dataset = ImageFolder(args.test_dir, transform=trf())
    print(test_dataset.class_to_idx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint)
    model.eval()
 
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签
    for i, (inputs, labels) in tqdm(enumerate(test_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
 
        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)
 
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
 
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], args.num_classes)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
 
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
 
    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in tqdm(range(args.num_classes)):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
 
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(args.num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.num_classes):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= args.num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
 
    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
 
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)
 
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(args.num_classes), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3_with_Confusion_Matrix/set113_roc.jpg')
    # plt.show()
 
 
if __name__ == '__main__':
    args = parser.parse_args()
    # 加载模型
    print(os.getcwd())
    MobileNetv3_small = MobileNetV3_small(num_classes=args.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MobileNetv3_small = MobileNetv3_small.to(device)
    test(MobileNetv3_small, args.weights_path)
