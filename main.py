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

def rho(X_batch: torch.Tensor, Y_batch: torch.Tensor,
        Y_sample: torch.Tensor, pi_matrix: torch.Tensor, kernel) -> torch.Tensor:
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

    return rho

class ReLUCNNGP(autograd.Function):
    @staticmethod
    def forward(ctx, xy, xx, yy):
        ctx.save_for_backward(xy, xx, yy)
        f32_tiny = np.finfo(np.float32).tiny
        xx_yy = xx*yy + f32_tiny
        eps = 1e-6
        # NOTE: Replaced rsqrt with 1/t.sqrt()+eps. Check with Prof For accuracy
        inverse_sqrt_xx_yy = 1 / (torch.sqrt(xx_yy) + eps)
        # Clamp these so the outputs are not NaN
        # Use small eps to avoid NaN during backpropagation
        cos_theta = (xy * inverse_sqrt_xx_yy).clamp(-1+eps, 1-eps)
        sin_theta = torch.sqrt((xx_yy - xy**2).clamp(min=eps))
        theta = torch.acos(cos_theta)
        xy = (sin_theta + (math.pi - theta)*xy) / (2*math.pi)
        return xy

    @staticmethod
    def backward(ctx, grad_output):
        xy, xx, yy = ctx.saved_tensors
        f32_tiny = np.finfo(np.float32).tiny
        xx_yy = xx*yy + f32_tiny
        eps = 1e-6
        # NOTE: See https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a*b+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a*b%29%29%29*c%29%2F%282pi%29+wrt+c for differentiation wrt xy
        # term_1 = xy / (torch.sqrt((xx_yy - xy**2).clamp(min=0)) + eps)
        # term_2 = xy / (torch.sqrt(xx_yy.clamp(min=0)) * torch.sqrt((1 - xy**2 / xx_yy).clamp(min=0)) + eps)
        # term_3 = torch.acos((xy / (torch.sqrt(xx_yy.clamp(min=0)) + eps)).clamp(-1, 1))
        term_1 = xy / (torch.sqrt((xx_yy - xy**2).clamp(min=0)))
        term_2 = xy / (torch.sqrt(xx_yy.clamp(min=0)) * torch.sqrt((1 - xy**2 / xx_yy).clamp(min=0)))
        term_3 = torch.acos((xy / (torch.sqrt(xx_yy.clamp(min=0)))).clamp(-1, 1))
        # Convert nans to 0 and inf to large numbers
        term_1 = torch.nan_to_num(term_1, 0.0)
        term_2 = torch.nan_to_num(term_2, 0.0)
        term_3 = torch.nan_to_num(term_3, 0.0)
        diff_xy = (- term_1 + term_2 - term_3 + math.pi) / (2*math.pi)
        diff_xy_chained = grad_output * diff_xy

        # NOTE: See https://www.wolframalpha.com/input?i=differentiate+%28sqrt%28a+-+c%5E2%29+%2B+%28pi+-+arccos%28c%2Fsqrt%28a%29%29%29*c%29%2F%282pi%29+wrt+a for differentiation wrt xx_yy.
        # term_1 = 1 / (2 * torch.sqrt((xx_yy - xy**2).clamp(min=0)) + eps)
        # term_2 = xy**2 / (2 * xx_yy**1.5 * torch.sqrt((1 - xy**2 / xx_yy).clamp(min=0)) + eps)
        term_1 = 1 / (2 * torch.sqrt((xx_yy - xy**2).clamp(min=0)))
        term_2 = xy**2 / (2 * xx_yy**1.5 * torch.sqrt((1 - xy**2 / xx_yy).clamp(min=0)))
        # Convert nans to 0 and inf to large numbers
        term_1 = torch.nan_to_num(term_1, 0.0)
        term_2 = torch.nan_to_num(term_2, 0.0)
        diff_xx_yy = (term_1 - term_2) / (2 * math.pi)

        diff_xx = diff_xx_yy * yy
        diff_xx_chained = grad_output * diff_xx
        diff_yy = diff_xx_yy * xx
        diff_yy_chained = grad_output * diff_yy

        # diff_xx_yy_chained = grad_output * diff_xx_yy

        return diff_xy_chained, diff_xx_chained, diff_yy_chained

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

    # K_xx = model(X_train[0:10], X_train[0:10])#, X_train[0:10])

    # pi_mat = KernelFlowsCNNGP.pi_matrix(sample_indices=np.array([1,3,5,7,9]), dimension=(5,10))
    # rho_val = rho(X_train[:10], Y_train[:10].to(torch.float32), Y_sample=Y_train[np.array([1,3,5,7,9])].to(torch.float32), pi_matrix=pi_mat, kernel=model)
    # rho_val.backward()
    # for params in model.parameters():
    #     print(params, params.grad)
