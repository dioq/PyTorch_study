#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from numba import cuda
import numpy as np
import math
from time import time


@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]


def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x
    gpu_result = np.zeros(n)
    cpu_result = np.zeros(n)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))
    if np.array_equal(cpu_result, gpu_result):
        print("result correct")


if __name__ == "__main__":
    main()
