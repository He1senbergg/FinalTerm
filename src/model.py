import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
from dataloader import cutmix
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vit_b_16, vgg11, ViT_B_16_Weights, VGG11_Weights

class vit_b16_expand_model(nn.Module):
    def __init__(self, pthpath: str = None, scratch: bool = False):
        super(vit_b16_expand_model, self).__init__()

        if pthpath or scratch:
            self.vit = vit_b_16(weights=None)
        else:
            # 加载预训练模型
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 100)  # 修改预训练模型中最后一个线性层的输出特征数为100
        
        # 获取现有的编码器层
        existing_layers = self.vit.encoder.layers
        
        # 复制最后六层，增加到模型中
        new_layers = [deepcopy(layer) for layer in existing_layers[-6:]]
        self.vit.encoder.layers.extend(new_layers)
        
        # 更新模型的层数
        self.vit.encoder.num_layers = len(self.vit.encoder.layers)

        if pthpath:
            checkpoint = torch.load(pthpath)
            self.vit.load_state_dict(checkpoint) 

    def forward(self, x):
        x = self.vit(x)
        return x

# 加载预训练的VGG11模型
class VGG_11(nn.Module):
    def __init__(self, pthpath: str = None, scratch: bool = False):
        super(VGG_11, self).__init__()  # 正确调用父类的构造函数
        if pthpath or scratch:
            self.vgg11 = vgg11(weights=None)
        else:
            self.vgg11 = vgg11(weights=VGG11_Weights.IMAGENET1K_V1) # 使用vgg11
        
        in_features = self.vgg11.classifier[-1].in_features
        self.vgg11.classifier[-1] = nn.Linear(in_features, 100)  # 调整输出类别为100

        if pthpath:
            checkpoint = torch.load(pthpath)
            self.vgg11.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.vgg11(x)  # 使用vgg11进行前向传播
        return x

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: Optimizer, num_epochs: int = 70, 
                logdir: str ='/mnt/ly/models/FinalTerm/mission2/tensorboard/1',
                save_dir: str ='/mnt/ly/models/FinalTerm/mission2/modelpth/1',
                milestones: list = None, gamma: float = 0.1):
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir=logdir)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # corrects = 0

        # 当optimizer为SGD时，更新学习率。
        # 当optimizer为Adam时，milestones为空，不会执行。
        if epoch+1 in milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma

        # 开始训练计时
        train_start_time = time.time()
        data_transform_time = 0.0
        for inputs, labels in train_loader:
            data_transform = time.time()
            inputs, labels1, labels2, lam = cutmix(inputs, labels, alpha=1.0)  # 应用 CutMix
            data_transform_time += time.time() - data_transform
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device) # 移动到设备
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels1) * lam + criterion(outputs, labels2) * (1. - lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            # cutmix计算准确率没什么意义，只需计算test accuracy
            # corrects += torch.sum(preds == labels.data)
            
            # total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        # cutmix计算准确率没什么意义，只需计算test accuracy
        # epoch_acc = corrects.double() / total

        # 结束训练计时
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f'Epoch {epoch+1}/{num_epochs}, \nTrain Loss: {epoch_loss:.4f}, \
              Data Transform Time: {data_transform_time: .2f}s, Training Time: {train_elapsed_time:.2f}s')

        # 将训练loss写入TensorBoard
        writer.add_scalar('Loss/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Data Transform Time', data_transform_time, epoch)
        writer.add_scalar('Time/Train', train_elapsed_time, epoch)

        # 将当前学习率写入TensorBoard
        lr_i = 1
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar(f'Learning Rate/{lr_i}', current_lr, epoch)
            lr_i += 1

        # 验证步骤
        model.eval()
        val_loss = 0.0
        corrects = 0

        # 开始验证计时
        val_start_time = time.time()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_end_time = time.time()
        val_elapsed_time = val_end_time - val_start_time

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test Time: {val_elapsed_time:.2f}s')

        # 将验证loss和accuracy写入TensorBoard
        writer.add_scalar('Loss/Test Loss', val_loss, epoch)
        writer.add_scalar('Accuracy/Test Accuracy', val_acc, epoch)
        writer.add_scalar('Time/Test', val_elapsed_time, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            file_path = f"{epoch+1}_{best_val_acc}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, file_path))
    
    writer.flush()
    writer.close()