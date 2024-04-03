# GPU 加速

在CUDA编程中，CPU和主存被称为主机（Host），GPU被称为设备（Device）

## CPU程序是顺序执行的，一般需要

初始化
CPU计算
得到计算结果

## 当引入GPU后，计算流程变为

初始化，并将必要的数据拷贝到GPU设备的显存上
CPU调用GPU函数，启动GPU多个核心同时进行计算
CPU与GPU异步计算
将GPU计算结果拷贝回主机端，得到计算结果

与传统的Python CPU代码不同的是：

使用from numba import cuda引入cuda库
在GPU函数上添加@cuda.jit装饰符，表示该函数是一个在GPU设备上运行的函数，GPU函数又被称为核函数。
主函数调用GPU核函数时，需要添加如[1, 2]这样的执行配置，这个配置是在告知GPU以多大的并行粒度同时进行计算。gpu_print[1, 2]()表示同时开启2个线程并行地执行gpu_print函数，函数将被并行地执行2次。下文会深入探讨如何设置执行配置。
GPU核函数的启动方式是异步的：启动GPU函数后，CPU不会等待GPU函数执行完毕才执行下一行代码。必要时，需要调用cuda.synchronize()，告知CPU等待GPU执行完核函数后，再进行CPU端后续计算。这个过程被称为同步，也就是GPU执行流程图中的红线部分。如果不调用cuda.synchronize()函数，执行结果也将改变，"print by cpu.将先被打印。虽然GPU函数在前，但是程序并没有等待GPU函数执行完，而是继续执行后面的cpu_print函数，由于CPU调用GPU有一定的延迟，反而后面的cpu_print先被执行，因此cpu_print的结果先被打印了出来。

## Thread层次结构

在进行GPU并行编程时需要定义执行配置来告知以怎样的方式去并行计算，比如上面打印的例子中，是并行地执行2次，还是8次，还是并行地执行20万次，或者2000万次。2000万的数字太大，远远多于GPU的核心数，如何将2000万次计算合理分配到所有GPU核心上。解决这些问题就需要弄明白CUDA的Thread层次结构。

CUDA将核函数所定义的运算称为线程（Thread），多个线程组成一个块（Block），多个块组成网格（Grid）。这样一个grid可以定义成千上万个线程，也就解决了并行执行上万次操作的问题。例如，把前面的程序改为并行执行8次：可以用2个block，每个block中有4个thread。原来的代码可以改为gpu_print[2, 4]()，其中方括号中第一个数字表示整个grid有多少个block，方括号中第二个数字表示一个block有多少个thread。

实际上，线程（thread）是一个编程上的软件概念。从硬件来看，thread运行在一个CUDA核心上，多个thread组成的block运行在Streaming Multiprocessor（SM的概念详见本系列第一篇文章），多个block组成的grid运行在一个GPU显卡上。

CUDA提供了一系列内置变量，以记录thread和block的大小及索引下标。以[2, 4]这样的配置为例：blockDim.x变量表示block的大小是4，即每个block有4个thread，threadIdx.x变量是一个从0到blockDim.x - 1（4-1=3）的索引下标，记录这是第几个thread；gridDim.x变量表示grid的大小是2，即每个grid有2个block，blockIdx.x变量是一个从0到gridDim.x - 1（2-1=1）的索引下标，记录这是第几个block。

某个thread在整个grid中的位置编号为：threadIdx.x + blockIdx.x * blockDim.x
