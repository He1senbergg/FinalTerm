import os
import torch
import argparse
import torch.nn as nn
from dataloader import get_loaders
from model import VGG_11, vit_b16_expand_model, train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT/VGG model on the CIFAR100 dataset.")
    parser.add_argument('--trytime', type=int, required=True, help='Try number for the training configuration.')
    parser.add_argument('--data_dir', type=str, default=r'/mnt/ly/models/FinalTerm/mission2/data', help='Path to the CIFAR100 dataset directory.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
    parser.add_argument('--pthpath', type=str, default=None, help='Path to a saved model checkpoint to continue training.')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD', help='Optimizer to use (SGD or Adam).')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for saving model and logs.')
    parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--milestones', type=list, default=[], help='List of epochs to decrease the learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Factor to decrease the learning rate.')
    parser.add_argument('--model', type=str, choices=['vgg11', 'vit'], default='vgg11', help='Model to train (VGG11 or ViT).')
    parser.add_argument('--scratch', type=bool, default=False, help='Train the model from scratch.')
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
    model_choice = args.model
    scratch = args.scratch
    optimizer= args.optimizer

    if optimizer == "SGD" and len(milestones) == 0:
        milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]

    if milestones:
        for milestone in milestones:
            if milestone > num_epochs:
                raise ValueError("Milestone epoch cannot be greater than the total number of epochs.")
            
    train_loader, test_loader = get_loaders(batch_size=batch_size, data_dir=data_dir)

    criterion = nn.CrossEntropyLoss()
    
    model = None
    if model_choice == "vgg11":
        if scratch or pthpath:
            model = VGG_11(pthpath=pthpath) if pthpath else VGG_11(scratch=True)
            parameters = [{"params": model.vgg11.parameters(), "lr": learning_rate}]
        else:
            model = VGG_11()
            parameters = [
                {"params": model.vgg11.classifier[-1].parameters(), "lr": learning_rate},  # 最后一个线性层
                {"params": [param for name, param in model.vgg11.named_parameters() if not name.startswith("classifier.6")], "lr": learning_rate * 0.1}  # 其余层，确保不包括最后一层
            ]       
    elif model_choice == "vit":
        if scratch or pthpath:
            model = vit_b16_expand_model(pthpath=pthpath) if pthpath else vit_b16_expand_model(scratch=True)
            parameters = [{"params": model.vit.parameters(), "lr": learning_rate}]
        else:
            model = vit_b16_expand_model()
            # 检索复制层的参数
            copied_layer_indices = [i for i in range(len(model.vit.encoder.layers) - 6, len(model.vit.encoder.layers))]
            parameters = [
                {"params": model.vit.heads.head.parameters(), "lr": learning_rate},  # 最后一个线性层
                {"params": [param for name, param in model.vit.named_parameters() if any(f"encoder.layers.{i}" in name for i in copied_layer_indices)], "lr": learning_rate},  # 复制的encoder层
                {"params": [param for name, param in model.vit.named_parameters() if "heads.head" not in name and all(f"encoder.layers.{i}" not in name for i in copied_layer_indices)], "lr": learning_rate * 0.1}  # 其余层
            ]
    else:
        raise ValueError("Model must be either 'vgg11' or 'vit'.")

    # 构造目录名称
    directory_name = f"{try_times}_{model_choice}_{optimizer}_{momentum}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{scratch}_{milestones}_{gamma}"
    
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
        optimizer = torch.optim.Adam(parameters, weight_decay=decay, eps=1e-3)

    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, logdir, save_dir, model_choice, milestones, gamma)

if __name__ == '__main__':
    main()