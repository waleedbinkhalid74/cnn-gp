import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')

from utils import get_MNIST_dataset, get_label_from_probability
from cnn_gp import NNGPKernel
from KernelFlow import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    X_train, Y_train, X_test, Y_test = get_MNIST_dataset(train_size=50000, val_size=1000, device=DEVICE)
    cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
    # Kernel Regression equation is as follows:
    # K(X', X) * (K(X,X)^-1 * Y) 
    # Where X' is the test dataset, X is the train dataset
    # and Y is the train targets

    # We will evaluate the accuracy for various sizes of N_i i.e. the number
    # of training points used for kernel regression
    N_i_arr = np.arange(100, 1600, 100)
    cpu_acc = []
    gpu_acc = [] 
    for N_i in tqdm(N_i_arr):
        X_train_Ni = X_train[:N_i]
        Y_train_Ni = Y_train[:N_i]
        X_test_Ni = X_test[:N_i]
        Y_test_Ni = Y_test[:N_i]

        with torch.no_grad():
            k_xx = get_kernel(X_train_Ni, X_train_Ni, cnngp=cnn_gp)
            k_x_prime_x = get_kernel(X_test_Ni, X_train_Ni, cnngp=cnn_gp)

        # CPU evaluation
        k_inv_Y_cpu = torch.linalg.lstsq(k_xx.cpu(), Y_train_Ni.cpu(), rcond=1e-8).solution
        prediction_cpu = torch.matmul(k_x_prime_x.cpu(), k_inv_Y_cpu)
        prediction_labels_cpu = get_label_from_probability(prediction_cpu)
        cpu_acc.append(accuracy_score(prediction_labels_cpu, Y_test_Ni.cpu().numpy()) * 100)

        # GPU evaluation
        k_inv_Y_gpu = torch.linalg.lstsq(k_xx, Y_train_Ni, rcond=1e-8).solution
        prediction_gpu = torch.matmul(k_x_prime_x, k_inv_Y_gpu)
        prediction_labels_gpu = get_label_from_probability(prediction_gpu)
        gpu_acc.append(accuracy_score(prediction_labels_gpu, Y_test_Ni.cpu().numpy()) * 100)

    fig, ax = plt.subplots(1,1)
    ax.plot(N_i_arr, cpu_acc, '-o', label="Kernel Regression on CPU")
    ax.plot(N_i_arr, gpu_acc, '-*', label="Kernel Regression on GPU")
    ax.set_xlabel("Number of datapoints used for Kernel Regression: $N_I$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,100))
    plt.legend()
    fig.savefig('./figs/cpu_vs_gpu_accuracy.png')
    plt.show()

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "covnet",
                    "which CNNGP model to test on")
    absl.app.run(main)


