from re import S
from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import torch.nn.functional as F
import os
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import frechet_grad, batch_creation
import pytest
from torchvision import datasets, transforms
import numpy as np
from KernelFlow.Frechet.kernel_functions import kernel_RBF
from kernels import RBF_Kernel

# def test_kernel_evaluation():
#     """Sanity check to ensure if reproducability exists upon changes made to source code.
#     """
#     model = Sequential(Conv2d(kernel_size=3),
#                         ReLU(),
#                         Conv2d(kernel_size=3, stride=2),
#                         ReLU(),
#                         Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
#                     )

#     X_test = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_input.pt")
#     K_xx = model(X_test, X_test)
#     K_xx_compare = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_output.pt")
#     assert torch.equal(K_xx, K_xx_compare)


def test_batch_sample_creation():
    """Check if lengths of batch and samples match
    """
    X = torch.rand((100, 1, 2,2))
    batch_size = 50
    sample_proportion = 0.5
    samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=100,
                                                       batch_size=batch_size,
                                                       sample_proportion=sample_proportion)

    samples1, batches1 = batch_creation(X, batch_size=50, sample_proportion=0.5)

    assert len(batches) == batch_size
    assert len(samples) == batch_size * sample_proportion
    assert len(samples1) == len(samples)
    assert len(batches) == len(batches1)

    X_batch = X[batches]
    X_sample = X_batch[samples]

    assert X_batch.shape == (50,1,2,2)
    assert X_sample.shape == (25,1,2,2)

    X_batch1 = X[batches1]
    X_sample1 = X_batch1[samples1]

    assert X_batch1.shape == (50,1,2,2)
    assert X_sample1.shape == (25,1,2,2)

def test_batch_creation_error_handling():
    """Check if Exceptions are caught
    """
    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=12,
                                                        sample_proportion=0.5)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=3,
                                                        sample_proportion=1.1)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=3,
                                                        sample_proportion=-0.5)


def test_pi_matrix():
    sample_indices = np.array([0,2,7,8,9])
    N_c = 5
    N_f = 10
    pi_matrix_compare = torch.Tensor([[1,0,0,0,0,0,0,0,0,0],
                                   [0,0,1,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,1,0,0],
                                   [0,0,0,0,0,0,0,0,1,0],
                                   [0,0,0,0,0,0,0,0,0,1]])
    pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices,dimension=(N_c, N_f))
    print(pi_matrix_compare)
    print(pi_matrix)
    assert torch.equal(pi_matrix, pi_matrix_compare)

def test_rho():
    """Test if calculation of rho is correct compared to Darcy's implementation
    """
    model = Sequential(
                Conv2d(kernel_size=3, padding=0),
                ReLU(),
                )
    K = KernelFlowsCNNGP(cnn_gp_kernel=model)
    X = torch.ones((10, 1, 3,3), dtype=torch.float32)
    for i in range(X.shape[0]):
        X[i] = X[i] * i

    Y = torch.arange(0,10)
    Y = F.one_hot(Y, 10)
    Y = Y.to(torch.float32)

    samples = np.array([2,3,4])
    batches = np.array([1, 3, 4, 5, 7])


    N_f = len(batches)
    N_c = len(samples)
    X_batch = X[batches]
    X_sample = X[samples]
    Y_batch = Y[batches]
    Y_sample = Y[samples]

    pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=samples, dimension=(N_c, N_f))
    rho = K.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)

    ##### TEST COMPARISION #####
    pi_comp =  np.array([[0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]], dtype=float)

    K_xx = model(X_batch, X_batch)
    K_xx = K_xx.detach().numpy()
    sample_matrix = np.matmul(pi_comp, np.matmul(K_xx, np.transpose(pi_comp)))
    inverse_data = np.linalg.inv(K_xx + 0.0001 * np.identity(K_xx.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + 0.0001 * np.identity(sample_matrix.shape[0]))
    top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
    bottom = np.matmul(Y_batch.T, np.matmul(inverse_data, Y_batch))
    rho_comp = 1 - np.trace(top)/np.trace(bottom)
    # TODO: Check if a relative tolerance of just 0.01 is acceptable.
    # Pytorch and numpy result in slightly different results due to numerical reasons
    # Is this acceptable?
    assert np.isclose(rho.detach().numpy(), rho_comp, 1e-2)

def test_predict():
    """Test if the prediction is done correctly
    """
    pass

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
    X_train, Y_train = dataiter.next()
    Y_train = F.one_hot(Y_train, 10)

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()

    kernel_control = kernel_RBF(matrix_1=torch.flatten(X_train, 1, -1).detach().numpy().squeeze(),
                            matrix_2=torch.flatten(X_train, 1, -1).detach().numpy().squeeze(),
                            parameters=np.array([4.0]))
    rbf_kernel = RBF_Kernel(parameters=4.0)
    kernel_test = rbf_kernel(matrix_1=torch.flatten(X_train, 1, -1), matrix_2=torch.flatten(X_train, 1, -1))
    print(kernel_test.detach().numpy() - kernel_control)
    assert np.all(np.isclose(kernel_test.detach().numpy(), kernel_control))