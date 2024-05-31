# FinalTerm

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
