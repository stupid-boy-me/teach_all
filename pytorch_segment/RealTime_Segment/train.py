import argparse
import os
import time

import pandas as pd
import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import Cityscapes, palette
from models import ADSCNet, AGLNet, FDMobileNet

from models import list_available_models
print("可用模型列表:", list_available_models())
list_model = ['adscnet', 'enet', 'lednet', 'aglnet', 'bisenetv1', 'bisenetv2', 'canet', 'cfpnet', 'cgnet', 'contextnet', 'dabnet', 'ddrnet', 'dfanet', 'edanet', 'erfnet', 'esnet', 'espnet', 'espnetv2', 'fanet', 'farseenet', 'fastscnn', 'fddwnet', 'fpenet', 'fssnet', 'icnet', 'linknet', 'litehrnet', 'liteseg', 'mininet', 'mininetv2', 'ppliteseg', 'regseg', 'segnet', 'shelfnet', 'sqnet', 'stdc', 'swiftnet', 'fdmobilenet', 'smp']

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''
    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device = "cuda"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001,max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1),target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:,i,...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:,i,...]) ** self.gamma * target_onehot[:,i,...] * torch.log(pt[:,i,...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss


# train or val for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_time, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, binary, in data_bar:
            # data, binary = data.to(device), binary.to(device)
            data = data.to(device, dtype=torch.float32)
            binary = binary.to(device, dtype=torch.long)    
            torch.cuda.synchronize()
            start_time = time.time()
            out = net(data)
            prediction = torch.argmax(out, dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            loss = loss_criterion(out, binary)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(prediction == binary).item() / binary.numel() * data.size(0)

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} mPA: {:.2f}% FPS: {:.0f}'
                                     .format('Train' if is_train else 'Val', epoch, epochs, total_loss / total_num,
                                             total_correct / total_num * 100, total_num / total_time))

    return total_loss / total_num, total_correct / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentation Model') 

    parser.add_argument('--data_path', default='/algdata02/yiguo.huang/Data/cityscapes/', type=str,help='Data path for cityscapes dataset')
    
    parser.add_argument('--crop_h', default=384, type=int, help='Crop height for training images')
    parser.add_argument('--crop_w', default=672, type=int, help='Crop width for training images')
    parser.add_argument('--batch_size', default=12, type=int, help='Number of data for each batch to train')
    parser.add_argument('--save_step', default=5, type=int, help='Number of steps to save predicted results')
    parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_cpt', default='/algdata02/yiguo.huang/Algo_research_HYG/RealTime_Segment/cpk', type=str, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model_name', default='FDMobileNet', type=str, help='model name')
    # args parse
    # args parse
    args = parser.parse_args()
    data_path, crop_h, crop_w, save_cpt, model_name = args.data_path, args.crop_h, args.crop_w, args.save_cpt, args.model_name

    save_cpt = os.path.join(save_cpt, f"{model_name}_{crop_w}_{crop_h}")
    os.makedirs(save_cpt,exist_ok=True)
    batch_size, save_step, epochs = args.batch_size, args.save_step, args.epochs

    # dataset, model setup and optimizer config
    train_data      = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data        = Cityscapes(root=data_path, split='val',   crop_size=(crop_h, crop_w))
    train_loader    = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader      = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    # model = FastSCNN(in_channels=3, num_classes=2).cuda()
    model = FDMobileNet(num_classes=20,pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    loss_criterion = nn.CrossEntropyLoss(ignore_index=255)
    # loss_criterion = FocalLoss(gamma=2, alpha=[0.05, 0.25], n_class=2, device=device)

    # training loop
    results = {'train_loss': [], 'val_loss': [], 'train_mPA': [], 'val_mPA': []}
    best_mPA = 0.0
    for epoch in range(1, epochs + 1):
        print(f"使用的数据集是{data_path}")
        print(f"save的地址是:{save_cpt}")
        train_loss, train_mPA = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_mPA'].append(train_mPA)
        val_loss, val_mPA = train_val(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_mPA'].append(val_mPA)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(save_cpt, 'FastScnn{}_{}_statistics.csv'.format(int(crop_h ), int(crop_w ))), index_label='epoch')
        if val_mPA > best_mPA:
            best_mPA = val_mPA
            torch.save(model.state_dict(), os.path.join(save_cpt, 'FastScnn{}_{}_model.pth'.format(int(crop_h), int(crop_w))))
