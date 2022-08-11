# Parameter management

## 参数访问

- e.g. 单隐藏层的MLP

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

访问方式：

1. net.state_dict() 看整体（具体值）

![Untitled](Parameter%20management%20369a8b87cca34e56a0854b97312e75d8/Untitled.png)

1. 具体到某一层网络 net[n].state_dict()

![Untitled](Parameter%20management%20369a8b87cca34e56a0854b97312e75d8/Untitled%201.png)

1. 一次访问所有参数

```python
print(*[(name, param.shape)for name, paramin net[0].named_parameters()])
print(*[(name, param.shape)for name, paramin net.named_parameters()])
```

![Untitled](Parameter%20management%20369a8b87cca34e56a0854b97312e75d8/Untitled%202.png)

- 从嵌套块收集参数

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)
```

![Untitled](Parameter%20management%20369a8b87cca34e56a0854b97312e75d8/Untitled%203.png)

因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。 下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。`rgnet[0][1][0].bias.data`

## 初始化参数

- 内置初始化器
1. 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
```

1. 也可初始化为给定常数  

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
```

1. 分别应用不同的初始化方法（net还是之前定义的具有单隐藏层的MLP）

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) #Xavier初始化方法
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)     
```

1. 自定义初始化方法

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
```

1. 直接设置参数

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

![Untitled](Parameter%20management%20369a8b87cca34e56a0854b97312e75d8/Untitled%204.png)

1. 共享参数

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

共享参数通常可以节省内存，并在以下方面具有特定的好处：

- 对于图像识别中的CNN，共享参数使网络能够在图像中的任何地方而不是仅在某个区域中查找给定的功能。
- 对于RNN，它在序列的各个时间步之间共享参数，因此可以很好地推广到不同序列长度的示例。
- 对于自动编码器，编码器和解码器共享参数。 在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交。