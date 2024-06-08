import time
import torch
import argparse
import torch.nn as nn
from dataloader import get_loaders
from model import VGG_11, vit_b16_expand_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT/VGG model on the CIFAR100 dataset.")
    parser.add_argument('--data_dir', type=str, default=r'/mnt/ly/models/FinalTerm/mission2/data', help='Path to the CIFAR100 dataset directory.')
    parser.add_argument('--pthpath', type=str, required=True, help='Path to a saved model checkpoint to continue training.')
    parser.add_argument('--batch_size', type=int, required=True, help='The batch_size used during training.')
    parser.add_argument('--model', type=str, choices=['vgg11', 'vit'], required=True, help='The model to use for training (VGG11 or ViT).')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_dir = args.data_dir
    pthpath = args.pthpath
    batch_size = args.batch_size
    model = args.model

    train_loader, test_loader = get_loaders(batch_size, data_dir)

    criterion = nn.CrossEntropyLoss()

    if model == "vgg11":
        model = VGG_11(pthpath=pthpath) 
    elif model == "vit":
        model = vit_b16_expand_model(pthpath=pthpath)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 验证步骤
    model.eval()
    val_loss = 0.0
    corrects = 0

    # 开始验证计时
    val_start_time = time.time()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    val_end_time = time.time()
    val_elapsed_time = val_end_time - val_start_time

    val_loss = val_loss / len(test_loader.dataset)
    val_acc = corrects.double() / len(test_loader.dataset)
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Time: {val_elapsed_time:.2f}s')

if __name__ == '__main__':
    main()