#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

"""
ANN(人工神经网络)
使用多层神经元和激活函数来构成一个复杂的非线性映射函数.使用基于梯度的优化算法,可能陷入局部最优解
"""


# 定义一个神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = t.relu(self.layer1(x))
        return self.layer2(x)


def test01():
    x = np.arange(0, 20, dtype="float32")
    y = 2.5 * x + 5 * np.random.uniform(-1, 1)

    model = NeuralNetwork()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    x_tensor = t.from_numpy(x.reshape(-1, 1))  # 转换为张量
    y_tensor = t.from_numpy(y.reshape(-1, 1))
    for _ in range(1000):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # 预测新数据
    y_new = model(x_tensor)

    # 可视化
    plt.figure(figsize=(16, 9))
    plt.title("ANN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_new.detach().numpy(), color="red", label="predict")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test01()
