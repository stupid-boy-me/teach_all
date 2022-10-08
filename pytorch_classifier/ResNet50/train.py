import os
import sys
from model import ResNet
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from dataset import MyDataset
from utils import read_split_data
from tqdm import tqdm
'''
接下来的思路：
1.如何添加参数
2.如何可视化，且保存文件
3.如何动态调整学习率 train.py 
4.如何对self.stage3的后面添加一层其他网络，如FPN。其实也是更换backbone
5.对常见的激活函数进行学习，并且用自己的代码进行改进
6.多种优化器的选择
7.混合精度是什么意思？
8.如何计算loss和accuracy
9.如何画出混淆矩阵
10.使用json，cfg，yaml和args对网络进行代码规范
'''
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("we are using {} for training".format(device))

    transform = {
        "train":transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    data_root = r"E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\resnet\data\\"
    train_images_path,train_images_labels,val_images_path,val_images_labels = read_split_data(data_root)

    train_dataset = MyDataset(image_path=train_images_path,image_cla=train_images_labels,transform=transform["train"])
    val_dataset = MyDataset(image_path=val_images_path, image_cla=val_images_labels, transform=transform["val"])

    batch_size = 16
    nw = 1 #min(os.cpu_count(),batch_size if batch_size>1 else 0,8)
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
                                               # collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw)
                                               #collate_fn=val_dataset.collate_fn)


    model = ResNet(3).to(device)

    # 在这里我们不考虑预训练权重
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_acc = 0.0
    epochs = 10
    save_path = './resnet50.pth'
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0
        train_bar = tqdm(train_loader,file=sys.stdout)
        for i ,data in enumerate(train_bar):
            image, label = data
            optimizer.zero_grad()
            predict = model(image.to(device))
            loss = loss_function(predict,label.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss".format(epoch+1,epochs,loss)

        # eval
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader,file=sys.stdout)
            for data in val_bar:
                val_images,val_labels = data
                val_output = model(val_images.to(device)) # 图
                # print("val_output",val_output)
                # print("val_output.shape",val_output.shape) # torch.Size([16, 3]) 二维图
                predict_y = torch.max(val_output,dim=1)
                '''
                [1.3612, 1.2469, 0.6359, 1.0156, 0.9860, 1.2296, 1.4641, 0.5753, 1.2874,
                    0.6910, 0.7173, 0.7282, 0.6099, 0.5150, 0.7613, 0.3735])和
                    indices=tensor([2, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2]))
                '''
                # print("predict_y",predict_y)


                predict_y1 = torch.max(val_output, dim=1)[1] #提取索引
                # print("predict_y1", predict_y1)
                # tensor([2, 0, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2])


                acc += torch.eq(predict_y1,val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch+1,epochs)
                
            val_accurate = acc / len(val_dataset)
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / len(train_loader), val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(),save_path)

if __name__ == '__main__':
    main()




