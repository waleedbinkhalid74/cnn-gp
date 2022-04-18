from copyreg import pickle
from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import numpy as np
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import batch_creation
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import autograd
from utils import get_accuracy
import matplotlib.pyplot as plt
from KernelFlow.Frechet.kernel_functions import kernel_RBF
from kernels import RBF_Kernel
import torchvision.transforms as T
from torch import nn, optim
from __future__ import annotations

def dumb():
    return 10

if __name__ == "__main__":
    print(dumb())
