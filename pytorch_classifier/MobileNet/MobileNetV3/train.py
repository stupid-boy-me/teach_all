import os
import sys
import argparse
import logging
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.tensorboard import SummaryWriter
from Mobilenetv3 import MobileNetV3_large,MobileNetV3_small
import torch
from transform_processing import transform
import torch.optim as optim
from dataset import MyDataset
import random
import warnings
import json
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from utils import data_set_split,adjust_learning_rate,train_one_epoch,evaluate,plot_confusion_matrix
os.environ ["CUDA_VISIBLE_DEVICES"] = '0'
# 参考 https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
parser = argparse.ArgumentParser(description='Training with pytorch')
# 需要改动的参数
parser.add_argument("--dataset_type",default='My_data',type=str,help='type of dataset')
parser.add_argument("--root_datasets",default="/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals//",help='Dataset directory path')
parser.add_argument("--target_data_folder",default="/algdata01/yiguo.huang/hyg_deep_learning_me/Dataset/animals_output//",help='target_data_folder directory path')
parser.add_argument("--Save_Confusion_Matrix_folder",default="/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3/Confusion_Matrix_picture//",help='Dataset directory path')
parser.add_argument("--Save_weights_folder",default="/algdata01/yiguo.huang/hyg_deep_learning_me/MobileNet/MobileNetv3/weights_folder//",help='target_data_folder directory path')
parser.add_argument("--json_root",default="class_indices.json",help='Save_Confusion_Matrix_folder directory path')
parser.add_argument('--net', default="MobileNetV3_small",help="The network architecture, it can be MobileNetV1, MobileNetV2,or MobileNetV3.")
parser.add_argument('--batch_size', default=8, type=int,help='Batch size for training')
parser.add_argument('--num_classes', default=5, type=int,help='class_num for training')
parser.add_argument('--theta', default=1, type=int,help='theta for training')
parser.add_argument('--num_epochs', default=500, type=int,help='the number epochs')
# 不需要改动的参数
parser.add_argument('--balance_data', action='store_true',help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)   
parser.add_argument('--momentum', default=0.9, type=float,help='Momentum value for optim')
parser.add_argument('--gamma', default=0.1, type=float,help='Gamma update for SGD')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=8, type=int,help='Number of workers used in dataloading')
parser.add_argument('--use_cuda', default=True, type=bool,help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='1_model_newspaper_hecheng_ToDesk/', help='Directory for saving checkpoint models')
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--seed', default=10, type=int,help='seed for initializing training. ')
parser.add_argument('--class_label', default=[], type=list,help='seed for initializing training. ')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    args = parser.parse_args()
    logging.info(args) # 打印你的args

        # 加了这句话，会让种子失效
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        '''设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，
        为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。'''
        logging.info("Use Cuda.")


    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tb_writer = SummaryWriter()
    if args.net == 'MobileNetV3_small':
        create_net = MobileNetV3_small(num_classes=args.num_classes).to(device)
    elif args.net == 'MobileNetV3_large':
        create_net = MobileNetV3_large(num_classes=args.num_classes).to(device)
        
    train_images_path, train_images_labels, val_images_path, val_images_labels = data_set_split(args.root_datasets,args.target_data_folder)

    with open(args.json_root, "r") as f:
        class_indict = json.load(f)

    for key, values in class_indict.items():
        args.class_label .append(values)

    train_dataset = MyDataset(image_path=train_images_path, image_cla=train_images_labels, transform=transform["train"])
    val_dataset = MyDataset(image_path=val_images_path, image_cla=val_images_labels, transform=transform["val"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=val_dataset.collate_fn)
    pg = [p for p in create_net.parameters() if p.requires_grad]
    '''
    调整学习率：是基于在每一个epoch调整，同时学习率的调整要在优化器参数之后更新。
    "即先优化器更新，在学习率更新"
    '''
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    lf = adjust_learning_rate(args)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(args.num_epochs):
        # train
        mean_loss = train_one_epoch(model=create_net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc,metric = evaluate(model=create_net,
                       data_loader=val_loader,
                       device=device)

        f = plot_confusion_matrix(metric.confusion_matrix(),args.num_classes,args.class_label,[args.num_classes,args.num_classes])
        f.savefig(args.Save_Confusion_Matrix_folder + '/{}-{}.png'.format("epoch",epoch))
        # f.clf()
        plt.clf()
        f.clear()  # 释放内存
        plt.close()
        
        logging.info("[epoch {}] accuracy: {}".format(epoch, round(acc, 5)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if (epoch+1) % 5 == 0:
            torch.save(create_net.state_dict(), args.Save_weights_folder +"/model-{}-of-{}-{}-{}.pth".format(epoch+1,args.num_epochs,mean_loss,acc))

if __name__ == '__main__':
    main()
