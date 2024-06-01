## 1.下载相关依赖
```bash
pip install torch torchvision tensorboard numpy
```
## 2.数据集的准备
代码中已经包括了数据集下载部分，且都可以通过`torchvision.datasets`直接下载
## 3.实验
可以更换超参数进行自监督实验
```python
learning_rates = [5e-4, 1e-4]
batch_sizes = [256, 128]
subset_ratios = [1.0, 0.5]
```

```bash
python self-supervision.py
```
更换数据集进行自监督实验

```bash
python self-supervisioncifar10.py
```
用有监督预训练好的ResNet实验
```bash
python supervised.py
```
从头训练ResNet18
```bash
python supervised_from_scratch.py
```
运用以下命令行查看Tesnsorboard可视化效果
```bash
tensorboard --logdir=runs
```


"# DLNN_3_1" 
#   D L N N _ 3 _ 1  
 