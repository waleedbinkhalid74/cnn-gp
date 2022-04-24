from copyreg import pickle
import math
from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import numpy as np
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import batch_creation
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import autograd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch import autograd

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 10000
    val_size = 1000
    N_I = 1000

    # MNIST
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()

    # model = Sequential(
    #     Conv2d(kernel_size=3),
    #     ReLU(),
    #     Conv2d(kernel_size=3, stride=2),
    #     ReLU(),
    #     Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
    #     )

    var_bias = 7.86
    var_weight = 2.79

    layers = []
    for _ in range(7):  # n_layers
        layers += [
            Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
                var_bias=var_bias),
            ReLU(),
        ]
    model = Sequential(
        *layers,
        Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
            var_bias=var_bias),
    )

    # model = Sequential(
    #     Conv2d(kernel_size=28, padding=0, var_weight=1.0, var_bias=1.0),  # equivalent to a dense layer
    #     )

    # model = Sequential(
    #     Conv2d(kernel_size=28, padding=0, var_weight=1.0, var_bias=1.0),  # equivalent to a dense layer
    #     ReLU()
    #     )

    # model = Sequential(
    #     ReLU(),
    #     Conv2d(kernel_size=28, padding=0, var_weight=1.0, var_bias=1.0)  # equivalent to a dense layer
    #     )
    K_xx = model(X_train[0:10], X_train[0:10])#, X_train[0:10])
    print(K_xx)


