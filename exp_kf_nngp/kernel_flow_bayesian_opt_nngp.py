import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from jax import jit, grad
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch
from neural_tangents import stax
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from skopt.plots import plot_convergence
import sys
import os
sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsPJAX

def get_flat_mnist():
    batch_size = 50000
    Val_size = 1000
    test_size = 1000

    transform_train = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
    transform_val = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
    transform_test = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])

    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform_train)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_val)
    testset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=Val_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=True)

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    X_train = X_train.numpy()
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)
    Y_train = Y_train.numpy()

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()
    return X_train, Y_train, X_test, Y_test

def get_kernel(depth, activation = stax.Relu(), out_dims = 10):
    W_std = np.random.rand()*100
    b_std = np.random.rand()*100
    layers = []
    for i in range(depth - 1):
        layers += [stax.Dense(25, W_std=W_std, b_std=b_std), activation]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers,
                stax.Dense(out_dims, W_std=W_std, b_std=b_std))
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return kernel_fn

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_flat_mnist()
    N_i_arr = [100, 250, 500, 750, 1000, 1250, 1500]
    rand_acc_matrix = []
    for i in range(5):   
        rand_acc = []
        kernel_fn = get_kernel(3)
        for N_i in tqdm(N_i_arr):
            prediction = KernelFlowsPJAX.kernel_regression(kernel_fn, X_train=X_train[:N_i], 
                                                            Y_train=Y_train[:N_i], X_test=X_test)
            prediction_labels = np.argmax(prediction, axis=1)
            rand_acc.append(accuracy_score(prediction_labels, Y_test) * 100)
        rand_acc_matrix.append(np.array(rand_acc))

    rand_acc_matrix = np.array(rand_acc_matrix)
    rand_min = rand_acc_matrix.min(axis=0)
    rand_max = rand_acc_matrix.max(axis=0)

    parameter_bounds = [(0.25, 100.0), (0.0, 100)]
    KF_JAX = KernelFlowsPJAX(kernel_layers=3, kernel_output_dim=10)
    bo_res = KF_JAX.fit_bayesian_optimization(X=X_train, Y=Y_train, batch_size=1200, 
                                                iterations=50, random_starts=15, 
                                                parameter_bounds_BO=parameter_bounds)
    fig, ax = plt.subplots(1,1)
    plot_convergence(bo_res, ax=ax)
    ax.set_ylim((0,1))
    fig.savefig('figs/nngp_parametric_kf_bo-rho_convergence.png')

    trained_acc = []
    for N_i in tqdm(N_i_arr):
        prediction = KernelFlowsPJAX.kernel_regression(KF_JAX.optimized_kernel, X_train=X_train[:N_i], Y_train=Y_train[:N_i], X_test=X_test)
        prediction_labels = np.argmax(prediction, axis=1)
        trained_acc.append(accuracy_score(prediction_labels, Y_test) * 100)

    fig, ax = plt.subplots(1,1)
    ax.plot(N_i_arr, trained_acc, 'o-', label='Bayesian Optimization Trained NNGP')
    ax.fill_between(N_i_arr, rand_min, rand_max, alpha=0.25, label='NNGPs with randomly initialized $\sigma_w$ and $\sigma_b$')
    ax.set_ylim((0, 100))
    ax.set_xlabel("Number of input samples used for Kernel Regression $N_I$")
    ax.set_ylabel("Accuracy")
    plt.yticks(np.arange(0, 101, 5.0))
    plt.legend()
    fig.savefig('figs/nngp_parametric_kf_bo_accuracy.png')
