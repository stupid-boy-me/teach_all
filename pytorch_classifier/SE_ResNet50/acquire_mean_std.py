from __future__ import print_function, division
import torch
import torch.utils.data
import torchvision
import torchvision.datasets as Datasat
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    dataset = Datasat.ImageFolder(root=r"E:\Project\tiankong_datasets\script\deep_learning_me\pytorch_classification_me\Dataset\data_animals_ten\animals10\raw-img", transform=torchvision.transforms.ToTensor())
    print(dataset)
    print(getStat(dataset))