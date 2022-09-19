"""
This script demonstrates the differences that arise when performing kernel ridge regression on the CPU vs on the GPU under different precision settings on pytorch.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os
import time

sys.path.insert(0, os.getcwd() + '/.')

from utils import get_MNIST_dataset, get_label_from_probability
from cnn_gp import NNGPKernel
from KernelFlow import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_seed = int(time.time())
torch.manual_seed(time_seed)
# torch.manual_seed(1655044203)
# 1655044203
def get_kernel(X: torch.Tensor, Y: torch.Tensor, cnngp: NNGPKernel) -> torch.Tensor:
    """Evaluates the Kernel for given X and Y

    Args:
        X (torch.Tensor): X input to Kernel
        Y (torch.Tensor): Y input to Kernel
        cnngp (NNGPKernel): Callable Kernel function

    Returns:
        torch.Tensor: Evaluated Kernel
    """
    return KernelFlowsTorch._block_kernel_eval(X, Y, 200, cnngp, device=DEVICE)

def main(_):

    X_train, Y_train, X_test, Y_test = get_MNIST_dataset(train_size=50000, val_size=1000, device=DEVICE, shuffle=True)
    cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
    # Kernel Regression equation is as follows:
    # K(X', X) * (K(X,X)^-1 * Y) 
    # Where X' is the test dataset, X is the train dataset
    # and Y is the train targets

    # We will evaluate the accuracy for various sizes of N_i i.e. the number
    # of training points used for kernel regression
    N_i_arr = np.arange(100, 1100, 100)
    cpu_acc = []
    gpu_acc = [] 
    cpu_acc_float64 = []
    gpu_acc_float64 = []
    normed_diff_matmul_cpu = []
    normed_diff_matmul_gpu = []
    normed_diff_matmul_gpu64 = []
    for N_i in tqdm(N_i_arr):
        X_train_Ni = X_train[:N_i]
        Y_train_Ni = Y_train[:N_i]

        with torch.no_grad():
            k_xx = get_kernel(X_train_Ni, X_train_Ni, cnngp=cnn_gp)
            k_x_prime_x = get_kernel(X_test, X_train_Ni, cnngp=cnn_gp)

    #########################################################################################################
        # # CPU evaluation
        # k_inv_Y_cpu = torch.linalg.lstsq(k_xx.cpu(), Y_train_Ni.cpu(), rcond=1e-8).solution
        # # k_inv_Y_cpu = torch.matmul(torch.linalg.inv(k_xx.cpu()), Y_train_Ni.cpu())
        # prediction_cpu = torch.matmul(k_x_prime_x.cpu(), k_inv_Y_cpu)
        # prediction_labels_cpu = get_label_from_probability(prediction_cpu)
        # cpu_acc.append(accuracy_score(prediction_labels_cpu, Y_test.cpu().numpy()) * 100)

        # # GPU evaluation
        # k_inv_Y_gpu = torch.linalg.lstsq(k_xx, Y_train_Ni, rcond=1e-8).solution
        # # k_inv_Y_gpu = torch.matmul(torch.linalg.inv(k_xx), Y_train_Ni)
        # prediction_gpu = torch.matmul(k_x_prime_x, k_inv_Y_gpu)
        # prediction_labels_gpu = get_label_from_probability(prediction_gpu)
        # gpu_acc.append(accuracy_score(prediction_labels_gpu, Y_test.cpu().numpy()) * 100)

        # # CPU evaluation float 64
        # k_inv_Y_cpu_float64 = torch.linalg.lstsq(k_xx.cpu().to(torch.float64), Y_train_Ni.cpu().to(torch.float64), rcond=1e-8).solution
        # # k_inv_Y_cpu_float64 = torch.matmul(torch.linalg.inv(k_xx.cpu().to(torch.float64)), Y_train_Ni.cpu().to(torch.float64))
        # prediction_cpu_float64 = torch.matmul(k_x_prime_x.cpu().to(torch.float64), k_inv_Y_cpu_float64)
        # prediction_labels_cpu_float64 = get_label_from_probability(prediction_cpu_float64)
        # cpu_acc_float64.append(accuracy_score(prediction_labels_cpu_float64, Y_test.cpu().to(torch.float64).numpy()) * 100)

        # # GPU evaluation float 64
        # k_inv_Y_gpu_float64 = torch.linalg.lstsq(k_xx.to(torch.float64), Y_train_Ni.to(torch.float64), rcond=1e-8).solution
        # # k_inv_Y_gpu_float64 = torch.matmul(torch.linalg.inv(k_xx.to(torch.float64)), Y_train_Ni.to(torch.float64))
        # prediction_gpu_float64 = torch.matmul(k_x_prime_x.to(torch.float64), k_inv_Y_gpu_float64)
        # prediction_labels_gpu_float64 = get_label_from_probability(prediction_gpu_float64)
        # gpu_acc_float64.append(accuracy_score(prediction_labels_gpu_float64, Y_test.cpu().to(torch.float64).numpy()) * 100)
    #########################################################################################################
    
        k_inv_Y_cpu_float64 = torch.linalg.lstsq(k_xx.cpu().to(torch.float64), Y_train_Ni.cpu().to(torch.float64), rcond=1e-8).solution

        prediction_cpu = torch.matmul(k_x_prime_x.cpu(), k_inv_Y_cpu_float64.to(torch.float32))
        prediction_labels_cpu = get_label_from_probability(prediction_cpu)
        cpu_acc.append(accuracy_score(prediction_labels_cpu, Y_test.cpu().numpy()) * 100)

        prediction_gpu = torch.matmul(k_x_prime_x, k_inv_Y_cpu_float64.cuda().to(torch.float32))
        prediction_labels_gpu = get_label_from_probability(prediction_gpu)
        gpu_acc.append(accuracy_score(prediction_labels_gpu, Y_test.cpu().numpy()) * 100)

        prediction_cpu_float64 = torch.matmul(k_x_prime_x.cpu().to(torch.float64), k_inv_Y_cpu_float64)
        prediction_labels_cpu_float64 = get_label_from_probability(prediction_cpu_float64)
        cpu_acc_float64.append(accuracy_score(prediction_labels_cpu_float64, Y_test.cpu().to(torch.float64).numpy()) * 100)

        prediction_gpu_float64 = torch.matmul(k_x_prime_x.to(torch.float64), k_inv_Y_cpu_float64.cuda())
        prediction_labels_gpu_float64 = get_label_from_probability(prediction_gpu_float64)
        gpu_acc_float64.append(accuracy_score(prediction_labels_gpu_float64, Y_test.cpu().to(torch.float64).numpy()) * 100)

        normed_diff_matmul_cpu.append(torch.linalg.norm((prediction_cpu_float64 - prediction_cpu.to(torch.float64)) / prediction_cpu_float64))
        normed_diff_matmul_gpu.append(torch.linalg.norm((prediction_cpu_float64 - prediction_gpu.cpu().to(torch.float64)) / prediction_cpu_float64))
        normed_diff_matmul_gpu64.append(torch.linalg.norm((prediction_cpu_float64 - prediction_gpu_float64.cpu()) / prediction_cpu_float64))


    fig, ax = plt.subplots(1,1)
    ax.plot(N_i_arr, cpu_acc, '-o', label="Kernel Regression on CPU with 32 bit precision", alpha=0.35)
    ax.plot(N_i_arr, gpu_acc, '-*', label="Kernel Regression on GPU with 32 bit precision", alpha=0.35)
    ax.plot(N_i_arr, cpu_acc_float64, '-^', label="Kernel Regression on CPU with 64 bit precision", alpha=0.35)
    ax.plot(N_i_arr, gpu_acc_float64, '-^', label="Kernel Regression on GPU with 64 bit precision", alpha=0.35)
    ax.set_xlabel("Number of datapoints used for Kernel Regression: $N_I$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,100))
    ax.set_ylim((0,100))
    plt.legend(prop={'size': 6})
    fig.savefig('./figs/cpu_vs_gpu_kernel_regression_accuracy.png')

    fig, ax = plt.subplots(1,1)
    ax.semilogy(N_i_arr, normed_diff_matmul_cpu,  'o-', label='CPU 64 bit precision vs CPU 32 bit precision')
    ax.semilogy(N_i_arr, normed_diff_matmul_gpu, '*-',label='CPU 64 bit precision vs GPU 32 bit precision')
    ax.semilogy(N_i_arr, normed_diff_matmul_gpu64, '^-', label='CPU 64 bit precision vs GPU 64 bit precision')
    ax.set_xlabel("Number of datapoints used for Kernel Regression: $N_I$")
    ax.set_ylabel("$||CPU64(K_{xx^\prime}(K_{xx}^{-1}Y)_{CPU64}) - Other(K_{xx^\prime}(K_{xx}^{-1}Y)_{CPU64})||_F$")
    plt.legend(prop={'size': 6})
    fig.savefig('./figs/cpu_vs_gpu_kernel_regression_diff_norm.png')
    plt.show()

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "simple",
                    "which CNNGP model to test on")
    absl.app.run(main)


