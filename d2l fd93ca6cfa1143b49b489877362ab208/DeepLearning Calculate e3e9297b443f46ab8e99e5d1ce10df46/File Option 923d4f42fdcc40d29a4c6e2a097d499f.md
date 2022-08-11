# File Option

## 简单读写

- e.g.

```python
x = torch.arange(4)
#torch.save(x, 'x-file')
#x2 = torch.load('x-file')
y = torch.zeros(4)
torch.save([x, y],'xy-files')
x2, y2 = torch.load('xy-files')
(x2, y2)
```

![Untitled](File%20Option%20923d4f42fdcc40d29a4c6e2a097d499f/Untitled.png)

## 加载保存模型参数

- e.g.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

返回 （2*10） 的Tensor。

![Untitled](File%20Option%20923d4f42fdcc40d29a4c6e2a097d499f/Untitled%201.png)

将模型的参数存储在一个叫做“mlp.params”的文件中。

```python
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，我们实例化了原始多层感知机模型的一个备份。 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。

```python
cloneNet = MLP()
cloneNet.load_state_dict(torch.load('mlp.params'))
cloneNet.eval()
```

![Untitled](File%20Option%20923d4f42fdcc40d29a4c6e2a097d499f/Untitled%202.png)

由于两个实例具有相同的模型参数，在输入相同的`X`时， 两个实例的计算结果应该相同。

```python
Y_clone = cloneNet(X)
Y_clone == Y
```

![Untitled](File%20Option%20923d4f42fdcc40d29a4c6e2a097d499f/Untitled%203.png)