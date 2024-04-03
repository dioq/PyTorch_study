#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from numba import cuda

print(cuda.gpus)


def cpu_print():
    print("print by cpu.")


@cuda.jit
def gpu_print():
    # GPU核函数
    print("print by gpu.")


def main():
    gpu_print[1, 2]()
    cuda.synchronize()
    cpu_print()


if __name__ == "__main__":
    main()
