#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch as t
from matplotlib import pyplot as plt


# 构建input x和ture值,并为true添加一定的噪声
def target_data(x):
    t.manual_seed(1000)  # 设置pytorch随机数种子
    y = 3 * x**3 + 2 * x**2 + x + 4 + t.randn(x.size()) * 0.2
    return y


# 预测模型
def pred_mode(x, a0, a1, a2, a3):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3


def train(x, y):
    # 定义待求参数
    a0 = t.rand(1, requires_grad=True)
    a1 = t.rand(1, requires_grad=True)
    a2 = t.rand(1, requires_grad=True)
    a3 = t.rand(1, requires_grad=True)

    # 构建loss
    y_pred = pred_mode(x, a0, a1, a2, a3)
    loss = t.sum(0.5 * (y_pred - y) ** 2)

    # 迭代求参数
    lr = 0.001
    for _ in range(1000):
        y_pred = pred_mode(x, a0, a1, a2, a3)
        loss = t.sum(0.5 * (y_pred - y) ** 2)
        loss.backward()

        with t.no_grad():
            a0 -= lr * a0.grad
            a1 -= lr * a1.grad
            a2 -= lr * a2.grad
            a3 -= lr * a3.grad

            a0.grad.zero_()
            a1.grad.zero_()
            a2.grad.zero_()
            a3.grad.zero_()

    return a0, a1, a2, a3


if __name__ == "__main__":
    x = t.linspace(-1, 1, 100)
    y = target_data(x)

    # 训练模型
    a0, a1, a2, a3 = train(x, y)
    print(a0)
    print(a1)
    print(a2)
    print(a3)

    # 可视化
    plt.scatter(x, y)
    pre_y = pred_mode(x, a0, a1, a2, a3)
    plt.plot(x, pre_y.detach(), "b")
    plt.show()
