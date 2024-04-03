from __future__ import print_function
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

running_loss = 0.0
