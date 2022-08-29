from torchvision import datasets, transforms
import torch
from KernelFlow.Parametric.Frechet.kernel_functions import kernel_RBF
import numpy as np
from KernelFlow import KernelFlowsTorch
from KernelFlow.Torch.kernel_collection import RBF_Kernel
import torch.nn.functional as F

def test_rbf():
    transform = transforms.Compose([transforms.ToTensor()
                            ])

    batch_size = 10
    val_size = 100

    # MNIST
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataiter = iter(trainloader)
    X_train, Y_train = next(dataiter)
    Y_train = F.one_hot(Y_train, 10)

    dataiter_val = iter(valloader)
    X_test, Y_test = next(dataiter_val)

    kernel_control = kernel_RBF(matrix_1=torch.flatten(X_train, 1, -1).detach().numpy().squeeze(),
                            matrix_2=torch.flatten(X_train, 1, -1).detach().numpy().squeeze(),
                            parameters=np.array([4.0]))
    rbf_kernel = RBF_Kernel(parameters=4.0)
    kernel_test = rbf_kernel(matrix_1=torch.flatten(X_train, 1, -1), matrix_2=torch.flatten(X_train, 1, -1))
    print(kernel_test.detach().numpy() - kernel_control)
    assert np.all(np.isclose(kernel_test.detach().numpy(), kernel_control))
