import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size: int = 64, data_dir: str = r'/mnt/ly/models/FinalTerm/mission2/data'):
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为224x224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载数据集
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)  # Beta分布生成lambda
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)  # 生成随机裁剪区域

    new_data = data.clone()
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]  # 应用裁剪区域
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))  # 调整lambda为实际裁剪面积比例

    return new_data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2