import os
import torch
import argparse
import torch.nn as nn
from dataloader import Tiny_ImageNet, CIFAR100
from model import load_model, NTXentLoss, self_supervised_train, supervised_train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet-18 model on Tiny ImageNet or CIFAR-100.")
    parser.add_argument('--trytime', type=int, required=True, help='Try number for the training configuration.')
    parser.add_argument('--data_dir', type=str, default=r'/mnt/ly/models/FinalTerm/mission2/data', help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
    parser.add_argument('--pthpath', type=str, default=None, help='Path to a saved model checkpoint to continue training.')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', "AdamW"], default='SGD', help='Optimizer to use (SGD or Adam).')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for saving model and logs.')
    parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--milestones', type=list, default=[], help='List of epochs to decrease the learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Factor to decrease the learning rate.')
    parser.add_argument('--strategy', type=str, choices=['ss', "s", 'sl', "pl"], required=True, help='Self-Supervised or Supervised or Self-Supervised Linear Protocal or Pretrained Linear Protocal.')
    return parser.parse_args()

def main():
    args = parse_args()

    try_times = args.trytime
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    momentum = args.momentum
    pthpath = args.pthpath
    base_dir = args.base_dir
    decay = args.decay
    milestones = args.milestones
    gamma = args.gamma
    strategy = args.strategy
    optimizer= args.optimizer

    if optimizer == "SGD":
        if len(milestones) == 0:
            milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]
        else:
            milestones = [int(x) for x in "".join(milestones[1:-1]).split(',')]

    if len(milestones) > 0:
        for milestone in milestones:
            if milestone > num_epochs:
                raise ValueError("Milestone epoch cannot be greater than the total number of epochs.")
    
    # 如果是自监督学习
    if strategy == "ss":       
        # 加载Tiny ImageNet数据集 
        train_loader = Tiny_ImageNet(batch_size=batch_size, data_dir=data_dir)
        # 加载SimCLR算法对应的模型
        model = load_model(self_supervised=True)
        # 分离出所有层的参数
        parameters = [{"params": model.parameters(), "lr": learning_rate}]
        criterion = NTXentLoss()
    # 如果是自监督学习的评估
    elif strategy == "sl":
        # 加载CIFAR100数据集 
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        if not pthpath:
            raise ValueError("进行自监督学习的评估时，需要提供自监督学习得到的模型的路径。")
        # 加载SimCLR算法对应的模型去除projection head层后的模型。
        model = load_model(self_supervised=True, linear_protocal=True, pthpath=pthpath)  
        # 冻结模型除分类层以外的所有层的参数
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False 
        # 分离出分类层的参数
        parameters = [{'params': model.fc.parameters(), 'lr': learning_rate}]  
        criterion = nn.CrossEntropyLoss()   
    # 如果是从零开始的监督学习
    elif strategy == "s":
        # 加载CIFAR100数据集
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        model = load_model(supervised=True)
        parameters = [{"params": model.parameters(), "lr": learning_rate}]
        criterion = nn.CrossEntropyLoss()
    # 如果是加载torch中在ImageNet上预训练的模型进行评估
    elif strategy == "pl":
        # 加载CIFAR100数据集
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        model = load_model(pretrained=True)
        # 冻结模型除分类层以外的所有层的参数
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False 
        # 分离出分类层的参数
        parameters = [{"params": model.fc.parameters(), "lr": learning_rate}]
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Please provide the correct strategy.")

    # 构造目录名称
    directory_name = f"{try_times}_{strategy}_{optimizer}_{momentum}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{milestones}_{gamma}"
    
    # 设置 save_dir 和 logdir
    save_dir = os.path.join(base_dir, "modelpth", directory_name)
    logdir = os.path.join(base_dir, "tensorboard", directory_name)

    # 尝试创建目录，如果它们已存在，则抛出异常
    try:
        os.makedirs(save_dir)
        os.makedirs(logdir)
    except FileExistsError:
        raise Exception(
            "文件名重复，请检查是否在同一个trytime下使用了同一个训练配置。"
            "如仍需尝试该配置，请将上一次运行的保存文件夹进行重命名（如末尾加上'_0'）"
            "或者删除该文件夹（请确保是不需要了再删除）。"
        )

    # 根据命令行参数选择优化器
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, weight_decay=decay)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, weight_decay=decay, eps=1e-8)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, weight_decay=decay, eps=1e-8)

    if strategy == "ss":
        self_supervised_train(model, train_loader, optimizer, criterion, num_epochs, logdir, save_dir, milestones, gamma)
    elif strategy == "sl":
        supervised_train(model, train_loader, test_loader, optimizer, criterion, num_epochs, logdir, save_dir, milestones, gamma)
    elif strategy == "s":
        supervised_train(model, train_loader, test_loader, optimizer, criterion, num_epochs, logdir, save_dir, milestones, gamma)
    elif strategy == "pl":
        supervised_train(model, train_loader, test_loader, optimizer, criterion, num_epochs, logdir, save_dir, milestones, gamma)
    
if __name__ == '__main__':
    main()