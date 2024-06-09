# FinalTerm（2024/06/09 11: 00）

# 仓库文件说明
## MISSION1 对比监督学习和自监督学习在图像分类任务上的性能表现
一共四个代码文件。
- `dataloader.py`：导入数据。
- `model.py`：模型class实现、训练函数实现。
- `train.py`：主要调用python文件，在其中导入了`dataloader.py`与`model.py`。在使用时，该文件需要命令行输入所需的参数（后文会明确指明），随后运行即可。
- `test.py`：用来实现测试pth的正确率。

## MISSION2 在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型
一共四个代码文件。
- `dataloader.py`：导入数据、实现Cutmix。
- `model.py`：模型class实现、训练函数实现。
- `train.py`：主要调用python文件，在其中导入了`dataloader.py`与`model.py`。在使用时，该文件需要命令行输入所需的参数（后文会明确指明），随后运行即可。
- `test.py`：用来实现测试pth的正确率。

# MISSION1
## Ⅰ. 准备步骤
**1. 代码下载**
下载Repo中`/mission1`下的四个python文件，放在同级目录。

调整终端目录，以便train.py能方便的导入其他同级目录的函数。

命令行运行代码
```
cd 四个文件摆放的同级目录位置
```

**2. 数据集准备**
下载Tiny ImageNet

命令行运行代码（请注意修改以下的信息的绝对位置）
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P /path/to/your/directory
```

运行`data_process.py`，将数据处理为我的代码中，自监督学习所需的数据形式。

命令行运行代码（请注意修改以下的信息的绝对位置）
```
python data_process.py --data_dir /path/to/your/directory
```

**3. 可调参数概述**
| 参数名        | 类型    | 默认值                                         | 描述                                               |
|-------------|---------|-----------------------------------------------|----------------------------------------------------|
| `--trytime` | int     | (**必须指定**)                                     | 运行轮次序号         |
| `--data_dir`| str     | 服务器上的数据集位置(**必须指定**)      | Path to the CIFAR100 dataset directory.            |
| `--batch_size` | int  | 64                                             | Batch size for training.                           |
| `--num_epochs` | int  | 70                                             | 训练轮次设定                     |
| `--lr`      | float   | 0.001                                          | Learning rate for the optimizer.                   |
| `--momentum`| float   | 0.9                                            | Momentum for the SGD optimizer.                    |
| `--pthpath`| str     | None                                           | Path to a saved model checkpoint to continue training. |
| `--optimizer` | str  | 'SGD'(**大小写敏感**)                     | Optimizer to use (SGD or Adam or AdamW).                   |
| `--base_dir` | str   | (**必须指定**)                                     | Base directory for saving model and logs.          |
| `--decay`   | float  | 1e-3                                           | Weight decay for the optimizer.                    |
| `--milestones` | list | []                                          | List of epochs to decrease the learning rate.      |
| `--gamma`   | float  | 0.1                                            | Factor to decrease the learning rate.              |
| `--strategy`   | str    | (**必须指定，全小写缩写**)                 | Strategy for training.  |

**4. 必须自适应调整的参数**

`--try_times`: 运行轮次序号。

每次运行，都需要设置try_times。目的是为了辅助文件夹进行排序，其，实为放在开头的轮次序号。该数没有过多要求，只要是int且不在同一个int使用相同的配置即可（否则会自动弹出报错）。

`--data_dir`：改为本地CIFAR100的位置（绝对位置）

因为代码中的默认地址信息为服务器上的地址，所以本地运行时，必须在命令行中重新赋值以修改。

`--base_dir`：运行过程中，进行保存pathpth和log时的根目录。

根目录需要自适应修改。代码实现了，会在base_dir/model/下，以各个当前运行的参数进行命名文件夹名a，随后会在base_dir/model/a/下进行pth的保存。同理会在base_dir/tensorboard/a/下进行log的保存。

*注：其中文件夹名a为f"{try_times}_{model_choice}_{optimizer}_{momentum}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{scratch}_{milestones}_{gamma}"*

`--strategy`：训练策略

参数待选项含义
- "ss"：Self-Supervised (from scratch)
- "s"：Supervised (from scratch)
- "sl"：Self-supervised Linear-protocal (frozen the parameters before the FC layer)
- "pl"：Pretrain (on ImageNet via supervised) Linear-protocal (frozen the parameters before the FC layer)

**5. 下载模型权重文件**

模型权重1: 在ImageNet上pre-trained的ResNet-18进行线性评估训练得到的模型。
```
wget 
```

模型权重2: 从零经过自监督学习与线性评估得到的模型。
```
wget 
```

模型权重3：从零进行监督学习得到的模型。
```
wget 
```

## Ⅱ. 训练
待补全
```
python src/train.py --base_dir "/mnt/ly/models/FinalTerm/mission1/" --num_epochs 300 --data_dir "/mnt/ly/models/FinalTerm/mission1/dataset/tiny-imagenet-200/" --strategy ss --trytime 3
```

## Ⅲ. 测试
待补全

# MISSION2
## cutmix说明
1. 代码实现所在具体位置

/mission2/dataloader.py

![image](https://github.com/He1senbergg/FinalTerm-Part2/assets/148076707/4b33d7de-fae2-475e-9e0e-e4bfb84281a7)

2. 训练中调用的位置

/mission2/train.py中调用model.py

/mission2/model.py

![image](https://github.com/He1senbergg/FinalTerm/assets/148076707/ff4c59f4-479f-4d33-879a-335622c00f08)


## Ⅰ. 准备步骤
**1. 代码下载**
下载Repo中`/mission2`下的四个python文件，放在同级目录。

调整终端目录，以便train.py能方便的导入其他同级目录的函数。

命令行运行代码
```
cd 四个文件摆放的同级目录位置
```

**2. 可调参数概述**
| 参数名        | 类型    | 默认值                                         | 描述                                               |
|-------------|---------|-----------------------------------------------|----------------------------------------------------|
| `--trytime` | int     | (**必须指定**)                                     | 运行轮次序号         |
| `--data_dir`| str     | 服务器上的数据集位置(**必须指定**)      | Path to the CIFAR100 dataset directory.            |
| `--batch_size` | int  | 64                                             | Batch size for training.                           |
| `--num_epochs` | int  | 70                                             | 训练轮次设定                     |
| `--lr`      | float   | 0.001                                          | Learning rate for the optimizer.                   |
| `--momentum`| float   | 0.9                                            | Momentum for the SGD optimizer.                    |
| `--pthpath`| str     | None                                           | Path to a saved model checkpoint to continue training. |
| `--optimizer` | str  | 'SGD'(**大小写敏感**)                     | Optimizer to use (SGD or Adam or AdamW).                   |
| `--base_dir` | str   | (**必须指定**)                                     | Base directory for saving model and logs.          |
| `--decay`   | float  | 1e-3                                           | Weight decay for the optimizer.                    |
| `--milestones` | list | []                                          | List of epochs to decrease the learning rate.      |
| `--gamma`   | float  | 0.1                                            | Factor to decrease the learning rate.              |
| `--model`   | str    | 'vgg11'                                          | Model to train ("vgg11" or "vit").                     |
| `--scratch` | bool   | False                                          | Train the model from scratch.                      |

**3. 必须自适应调整的参数**

`--try_times`: 运行轮次序号。

每次运行，都需要设置try_times。目的是为了辅助文件夹进行排序，其，实为放在开头的轮次序号。该数没有过多要求，只要是int且不在同一个int使用相同的配置即可（否则会自动弹出报错）。

`--data_dir`：改为本地CIFAR100的位置（绝对位置）

因为代码中的默认地址信息为服务器上的地址，所以本地运行时，必须在命令行中重新赋值以修改。

`--base_dir`：运行过程中，进行保存pathpth和log时的根目录。

根目录需要自适应修改。代码实现了，会在base_dir/model/下，以各个当前运行的参数进行命名文件夹名a，随后会在base_dir/model/a/下进行pth的保存。同理会在base_dir/tensorboard/a/下进行log的保存。

*注：其中文件夹名a为f"{try_times}_{model_choice}_{optimizer}_{momentum}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{scratch}_{milestones}_{gamma}"*


**4. 下载模型权重文件**

模型权重1: 在pre-trained的ViT基础上微调得到的结果。(dropbox)
```
wget https://www.dropbox.com/scl/fi/n6nvljix73xyvpiih4b8i/183_0.9032.pth?rlkey=pjckvuv6kwg2clhuh10t60gvj&st=whxrahg8&dl=1
```

模型权重2: 在pre-trained的VGG-11基础上微调得到的结果。(Google Drive)

浏览器打开链接以后，进行pth的下载。
```
https://drive.google.com/file/d/1hoQ3OmsZ_wjpgozheug1yFJB09uAnb_B/view?usp=sharing 
```

## Ⅱ. 训练

命令行运行代码

- 示例1（请注意修改以下的信息的绝对位置）：使用预训练vgg11模型与默认参数进行训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --trytime 1 --num_epochs 10
  ```
- 示例2（请注意修改以下的信息的绝对位置）：使用预训练vit模型与默认参数进行训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --trytime 1 --num_epochs 10 --model vit
  ```
- 示例3（请注意修改以下的信息的绝对位置）：使用预训练vgg11模型、Adam与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --optimizer Adam --trytime 2 --num_epochs 10
  ```
- 示例4（请注意修改以下的信息的绝对位置）：使用随机初始化vgg11模型与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2--scratch True --trytime 3 --num_epochs 10
  ```
- 示例5（请注意修改以下的信息的绝对位置）：使用本地vgg11模型的pth与其他默认参数开始训练
  ```
  python train.py --data_dir /mission2/data --base_dir /mission2 --pthpath model.pth --trytime 4 --num_epochs 10
  ```

## Ⅲ. 测试

测试的效果为输出如下信息：
```python
print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Time: {val_elapsed_time:.2f}s')
```

测试时，必须提供四个参数'--data_dir'、'--pthpath'、'--batch_size'、'--model'。

命令行运行代码，示例如下（请注意修改以下的信息的绝对位置）：
```
python test.py --data_dir /mission2/data --pthpath model.pth --batch_size 64 --model vgg
```
