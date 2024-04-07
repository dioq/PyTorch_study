#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch as t
import numpy as np

"""
Tensor是PyTorch中重要的数据结构,可认为是一个高维数组。它可以是一个数(标量)、一维数组(向量)、二维数组(矩阵)以及更高维的数组。
Tensor 和 Numpy 的ndarrays类似, 但Tensor可以使用GPU进行加速。
"""


def test01():
    # 构建 5x3 矩阵,只是分配了空间,未初始化
    x = t.Tensor(5, 3)
    print(x)

    # 使用[0,1]均匀分布随机初始化二维数组
    x = t.rand(5, 3)
    print(x)

    print(x.size())  # 查看x的形状

    # 查看列的个数, 两种写法等价 x.size()[1], x.size(1)
    print(x.size(1))

    return x


def test02():
    x = t.rand(5, 3)
    print(x)
    y = t.rand(5, 3)
    print(y)
    # 加法的第一种写法
    z = x + y
    print(z)


def test03():
    x = t.rand(5, 3)
    print(x)
    y = t.rand(5, 3)
    print(y)

    # 加法的第二种写法
    z = t.add(x, y)
    print(z)

    result = t.Tensor(5, 3)  # 预先分配空间
    t.add(x, y, out=result)  # 输入到result
    print(result)


def test04():
    x = t.rand(5, 3)
    # print(x)
    y = t.rand(5, 3)
    print("最初y")
    print(y)

    print("第一种加法,y的结果")
    y.add(x)  # 普通加法,不改变y的内容
    print(y)

    """
    注意,函数名后面带下划线_ 的函数会修改Tensor本身。例如,x.add_(y)和x.t_()会改变 x,但x.add(y)和x.t()返回一个新的Tensor, 而x不变。
    """
    print("第二种加法,y的结果")
    y.add_(x)  # inplace 加法,y变了
    print(y)


def test05():
    # Tensor 和 Numpy的数组之间的互操作非常容易且快速。对于Tensor不支持的操作,可以先转为Numpy数组处理,之后再转回Tensor。
    a = t.ones(5)  # 新建一个全1的Tensor
    print(a)

    b = a.numpy()  # Tensor -> Numpy
    print(b)


def test06():
    """
    Tensor和numpy对象共享内存,所以他们之间的转换很快,而且几乎不会消耗什么资源。但这也意味着,如果其中一个变了,另外一个也会随之改变。
    """
    a = np.ones(5)
    b = t.from_numpy(a)  # Numpy->Tensor
    print(a)
    print(b)

    b.add_(1)  # 以`_`结尾的函数会修改自身
    print(a)
    print(b)  # Tensor和Numpy共享内存


def test07():
    x = t.rand(5, 3)
    print(x)
    y = t.rand(5, 3)
    print(y)
    # 在不支持CUDA的机器下,下一步不会运行
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    print(x + y)


if __name__ == "__main__":
    test07()
