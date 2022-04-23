from copyreg import pickle
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

def rho(X_batch: torch.Tensor, Y_batch: torch.Tensor,
        Y_sample: torch.Tensor, pi_matrix: torch.Tensor, kernel, nabla_matrix) -> torch.Tensor:
    """Calculates the rho which acts as the loss function for the Kernel Flow method. It evaluates how good the results were even when the
    training input was reduced by a factor.

    Args:
        X_batch (torch.Tensor): Training batch dataset
        Y_batch (torch.Tensor): Training batch dataset labels in one hot encoding
        Y_sample (torch.Tensor): Training sample dataset drawn from the batch in one hot encoding
        pi_matrix (torch.Tensor): pi matrix

    Returns:
        torch.Tensor: Resulting value of rho
    """
    # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)
    # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
    # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

    # Calculate kernel theta = Kernel(X_Nf, X_Nf). NOTE: This is the most expensive step of the algorithm
    theta = kernel(X_batch, X_batch)

    # Calculate sample_matrix = pi_mat*theta*pi_mat^T
    sample_matrix = torch.matmul(pi_matrix, torch.matmul(theta, torch.transpose(pi_matrix, 0, 1)))

    # Add regularization
    inverse_data = torch.linalg.inv(theta)# + 0 * torch.eye(theta.shape[0]))

    # Delete theta matrix to free memory as it is not needed beyond this point
    # del theta

    inverse_sample = torch.linalg.inv(sample_matrix)# + 0 * torch.eye(sample_matrix.shape[0]))

    # Calculate numerator
    numerator = torch.matmul(torch.transpose(Y_sample,0,1), torch.matmul(inverse_sample, Y_sample))
    # Calculate denominator
    denominator = torch.matmul(torch.transpose(Y_batch,0,1), torch.matmul(inverse_data, Y_batch))
    # Calculate rho
    rho = 1 - torch.trace(numerator)/torch.trace(denominator)

    inverse_data = inverse_data.detach().numpy()
    Y_batch = Y_batch.detach().numpy()
    pi_matrix = pi_matrix.detach().numpy()
    inverse_sample = inverse_sample.detach().numpy()
    denominator = denominator.detach().numpy()

    y_hat = np.matmul(inverse_data, Y_batch)
    z_hat = np.matmul(np.transpose(pi_matrix), np.matmul(inverse_sample, np.matmul(pi_matrix, Y_batch)))
    gradient = (1-rho.detach().numpy())*np.matmul(np.transpose(y_hat), np.matmul(nabla_matrix,y_hat)) - np.matmul(np.transpose(z_hat), np.matmul(nabla_matrix, z_hat))
    gradient = np.trace(gradient, axis1 = -2, axis2 =-1)
    gradient = -gradient/np.trace(denominator)
    gradient = np.squeeze(gradient)

    return rho, gradient #, sample_matrix, inverse_data, inverse_sample, numerator, denominator

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

    model = Sequential(
        Conv2d(kernel_size=28, padding=0, var_weight=1.0, var_bias=1.0),  # equivalent to a dense layer
        ReLU()
        )

    K_xx = model(X_train[0:10], X_train[0:10])#, X_train[0:10])
    # print(K_xx)
    # with torch.no_grad():
    K_xx_grad = model.backprop()

    # for params in model.parameters():
    #     print(params, params.grad)

    r = torch.rand(K_xx.shape[0],1)
    rKr = torch.matmul(r.transpose(0, 1), torch.matmul(K_xx*K_xx, r))
    rKr.backward()
    rK_primer = 2*torch.matmul(r.transpose(0, 1), torch.matmul(torch.tensor(K_xx_grad)*K_xx, r))
    print(rK_primer)

    # K_xx_grad = torch.tensor(K_xx_grad)
    # K_xx_grad = K_xx_grad[2:]

    # pi_mat = KernelFlowsCNNGP.pi_matrix(sample_indices=np.array([1,3,5,7,9]), dimension=(5,10))
    # rho_val, gradient = rho(X_train[:10], Y_train[:10].to(torch.float32), Y_sample=Y_train[np.array([1,3,5,7,9])].to(torch.float32), pi_matrix=pi_mat, kernel=model, nabla_matrix=K_xx_grad)
    # rho_val.backward()
    # print(gradient)
    for params in model.parameters():
        print(params, params.grad)
