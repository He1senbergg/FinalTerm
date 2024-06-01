# FinalTerm-Part2（初稿Markdown, 2024/06/01 11: 04）

## 模型论文链接
[ViT](https://arxiv.org/pdf/2010.11929)、[VGG](https://arxiv.org/pdf/1409.1556)

## cutmix说明
1. 代码实现所在具体位置

/src/dataloader.py

![image](https://github.com/He1senbergg/FinalTerm-Part2/assets/148076707/4b33d7de-fae2-475e-9e0e-e4bfb84281a7)

2. 训练中调用的位置

/src/model.py

![image](https://github.com/He1senbergg/FinalTerm-Part2/assets/148076707/641b16cf-29fc-4fa9-91e2-c4656108f5a5)

## Ⅰ. 准备步骤
**1. 代码下载**
下载Repo中的四个python文件，放在同级目录

调整终端目录，以便train.py能方便的导入其他同级目录的函数。

命令行运行代码
```
cd 四个文件摆放的同级目录位置
```

**2. 可调参数概述**
| 参数名        | 类型    | 默认值                                         | 描述                                               |
|-------------|---------|-----------------------------------------------|----------------------------------------------------|
| `--trytime` | int     | (必须指定)                                     | Try number for the training configuration.         |
| `--data_dir`| str     | '/mnt/ly/models/FinalTerm/mission2/data'       | Path to the CIFAR100 dataset directory.            |
| `--batch_size` | int  | 64                                             | Batch size for training.                           |
| `--num_epochs` | int  | 70                                             | Number of epochs for training.                     |
| `--lr`      | float   | 0.001                                          | Learning rate for the optimizer.                   |
| `--momentum`| float   | 0.9                                            | Momentum for the SGD optimizer.                    |
| `--pthpath`| str     | None                                           | Path to a saved model checkpoint to continue training. |
| `--optimizer` | str  | 'SGD'                                          | Optimizer to use (SGD or Adam).                    |
| `--base_dir` | str   | (必须指定)                                     | Base directory for saving model and logs.          |
| `--decay`   | float  | 1e-3                                           | Weight decay for the optimizer.                    |
| `--milestones` | list | None                                          | List of epochs to decrease the learning rate.      |
| `--gamma`   | float  | 0.1                                            | Factor to decrease the learning rate.              |
| `--model`   | str    | 'vgg'                                          | Model to train (VGG11 or ViT).                     |
| `--scratch` | bool   | False                                          | Train the model from scratch.                      |

**3. 必须自适应调整的参数**

因为代码中的默认地址信息为服务器上的地址，所以本地运行时，必须在命令行中重新赋值以修改。
- `--data_dir`：改为本地CIFAR100的位置（绝对位置）

根目录需要自适应修改。代码实现了，会在base_dir/model/下，以各个当前运行的参数进行命名文件夹名a，随后会在base_dir/model/a/下进行pth的保存。同理会在base_dir/tensorboard/a/下进行log的保存。

*注：其中文件夹名a为f"{try_times}_{model_choice}_{optimizer}_{momentum}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{scratch}_{milestones}_{gamma}"*
- `--base_dir`：运行过程中，进行保存pathpth和log时的根目录。

需要设置try_times，为了辅助文件夹进行排序，放在开头的轮次序号。该数没有过多要求，只要是int即可。
- `--try_times`: 运行轮次序号。

**4. 下载模型权重文件**

模型权重1: 在pre-trained的VGG-11基础上微调得到的结果。
```
wget 
```

模型权重2: 在pre-trained的ViT基础上微调得到的结果。
```
wget 
```

## Ⅱ. 训练

命令行运行代码

- 示例1（请注意修改以下的信息的绝对位置）：使用预训练模型与默认参数进行训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2
  ```
- 示例2（请注意修改以下的信息的绝对位置）：使用预训练模型、Adam与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --optimizer Adam
  ```
- 示例3（请注意修改以下的信息的绝对位置）：使用随机初始化与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2--scratch True
  ```
- 示例4（请注意修改以下的信息的绝对位置）：使用本地模型pth与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --pthpath model.pth
  ```

## Ⅲ. 测试

测试的效果为：输出如下信息
```python
print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Time: {val_elapsed_time:.2f}s')
```

测试时，需要提供两个参数'--data_dir'、'--pthpath'、'--batch_size'、'--model'。

命令行运行代码，示例如下（请注意修改以下的信息的绝对位置）：
```
python test.py --data_dir /mission2/data --pthpath model.pth --batch_size 64 --model vgg
```
