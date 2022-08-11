# 多输入输出通道

- 多输入通道

e.g. 具有两个输入通道的二维度互相关运算（核函数的输入通道数也为2，与输入通道数一致）

![简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。](%E5%A4%9A%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E9%80%9A%E9%81%93%20f6cd37daecc14ac3b89dafb8c534218e/Untitled.png)

简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。

多通道互相关运算的实现

```python
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```python
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

![Untitled](%E5%A4%9A%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E9%80%9A%E9%81%93%20f6cd37daecc14ac3b89dafb8c534218e/Untitled%201.png)

- 多输出通道

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

e.g. 

设计卷积核 K = torch.stack((K, K + 1, K + 2), 0)       torch.Size([3, 2, 2, 2])    3个2*2的核

![Untitled](%E5%A4%9A%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E9%80%9A%E9%81%93%20f6cd37daecc14ac3b89dafb8c534218e/Untitled%202.png)

```python
corr2d_multi_in_out(X, K)
```

![Untitled](%E5%A4%9A%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E9%80%9A%E9%81%93%20f6cd37daecc14ac3b89dafb8c534218e/Untitled%203.png)

- 1 * 1卷积

使用 **1×1卷积核** 与 **3个输入通道** 和 **2个输出通道** 的互相关计算

![输入通道是三个，所以核函数的输入通道数也是三个；此处定义核函数的输出通道数为2            将ci个输入值转换为co个输出值](%E5%A4%9A%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E9%80%9A%E9%81%93%20f6cd37daecc14ac3b89dafb8c534218e/Untitled%204.png)

输入通道是三个，所以核函数的输入通道数也是三个；此处定义核函数的输出通道数为2            将ci个输入值转换为co个输出值

使用全连接层测试实现1 * 1 卷积

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

```python
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```