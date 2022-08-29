import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP_ODE#, KernelFlowsNP, KernelFlowsNPAutograd, KernelFlowsNP_Autograd_ODE
from KernelFlow.Non_Parametric.ODE_based.kernel_functions import kernel_RBF

class MultiScaleSineWave(Dataset):

    def __init__(self, n=100, random=False):
        xbound_low = -np.pi
        xbound_high = np.pi
        # xbound_low = -1
        # xbound_high = 1
        # target_fn = lambda x: np.sin(x) + 0.2*np.sin(10*x) + 0.1*np.sin(30*x)
        # target_fn = lambda x: np.sin(x) + 0.5*np.sin(x+1) + 0.25*np.sin(x+2)
        # target_fn = lambda x: np.exp(-0.5*x)*np.sin(3*x)
        target_fn = lambda x: np.sin(3*x)
        # target_fn = lambda x: np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x)
        # f = np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x)
        # target_fn = lambda x: (np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x) - np.min(np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x))) / (np.max(np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x)) - np.min(np.sin(2*x)+3*np.sin(3*x)+2*np.sin(4*x)))
        # target_fn = lambda x: x
        if random:
            x = xbound_low + np.random.rand(n,1) * xbound_high*2
        else:
            x = np.linspace(xbound_low, xbound_high, n)
            x = np.reshape(x, (x.shape[0], -1))
        y = target_fn(x)
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)
        self.n_samples = n

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n_samples

def get_raw_kernel_regression_result(kernel_parameter: float, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, regression: float) -> float:
    mu = [kernel_parameter]
    K_xx = kernel_RBF(train_x, train_x, mu)
    K_xx += regression*np.eye(K_xx.shape[0])
    T_xx = kernel_RBF(test_x, train_x, mu)
    K_xx_Y = np.linalg.lstsq(K_xx, train_y, rcond=1e-8)[0]
    test_pred = np.matmul(T_xx, K_xx_Y)
    fig, ax = plt.subplots(1,1)
    ax.scatter(test_x, test_y, label='Function to fit')
    ax.scatter(test_x, test_pred, label='Prediction from Kernel Ridge Regression')
    plt.legend()
    plt.savefig("kernel_regression_result.png")
    return mean_squared_error(test_y, test_pred)

def get_np_kernel_flows_results(kernel_parameter: float, train_x, train_y, test_x, test_y, batch_size, iterations):
    mu = np.array([kernel_parameter])
    kernel_name = "RBF"
    KF_rbf = KernelFlowsNP_ODE(kernel_name, mu)
    iterations = 1000
    batch_size = 64
    X_train_perturbed = KF_rbf.fit(train_x, train_y, iterations, batch_size = batch_size)

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax[1].scatter(X_train_perturbed.reshape(-1,1), train_y)
    ax[1].scatter(train_x, train_y)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("f(x)")

    ax[0].scatter(X_train_perturbed.reshape(-1,1), train_x)
    ax[0].scatter(train_x, train_x)
    ax[0].set_xlabel('X original')
    ax[0].set_ylabel('X perturbed')
    plt.legend()
    plt.savefig("perturbations_from_nonparam_ode_kernel_flows.png")

    fig, ax = plt.subplots(1,1)
    test_pred = KF_rbf.predict(test_x)
    ax.scatter(test_x, test_pred, label='Prediction from Non-Parametric Kernel Flows Trained Kernel Regression')
    ax.scatter(test_x, test_y, label='Function to fit')
    plt.legend()
    plt.savefig("kernelFlow_kernel_regression_result.png")

    return mean_squared_error(test_y, test_pred)

if __name__ == "__main__":
    dataset = MultiScaleSineWave(80)
    dataset_loader = DataLoader(dataset=dataset, batch_size = 80, shuffle=True)
    train_x, train_y = next(iter(dataset_loader))


    dataset = MultiScaleSineWave(500)
    dataset_loader = DataLoader(dataset=dataset, batch_size = 500, shuffle=True)
    test_x, test_y = next(iter(dataset_loader))

    train_x = train_x.numpy()
    train_y = train_y.numpy()
    test_x = test_x.numpy()
    test_y = test_y.numpy()

    raw_kernel_regression_mse = get_raw_kernel_regression_result(2.5, train_x, train_y, test_x, test_y, regression = 1e-5)
    kernel_flows_kernel_regression_mse = get_np_kernel_flows_results(2.5, train_x, train_y, test_x, test_y, 64, 1000)
    print(raw_kernel_regression_mse, kernel_flows_kernel_regression_mse)