#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch as t
from torch.autograd import Variable

"""
Autograd: 自动微分
深度学习的算法本质上是通过反向传播求导数,而PyTorch的Autograd模块则实现了此功能
在Tensor上的所有操作,Autograd都能为它们自动提供微分,避免了手动计算导数的复杂过程
autograd.Variable是Autograd中的核心类,它简单封装了Tensor,并支持几乎所有Tensor有的操作
Tensor在被封装为Variable之后,可以调用它的.backward实现反向传播,自动计算所有梯度

Variable主要包含三个属性
data:       保存Variable所包含的Tensor
grad:       保存data对应的梯度,grad也是个Variable,而不是Tensor,它和data的形状一样
grad_fn:    指向一个Function对象,这个Function用来反向传播计算输入的梯度
"""


def test01():
    # 使用Tensor新建一个Variable
    x = Variable(t.ones(2, 2), requires_grad=True)
    # print(x)
    y = x.sum()
    print(y)

    tmp = y.grad_fn
    # print(tmp)

    tmp = y.backward()  # 反向传播,计算梯度
    # print(tmp)

    # 注意:grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
    tmp = x.grad
    print(tmp)


if __name__ == "__main__":
    test01()
