import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsPJAX
from helpers import *

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_flat_mnist()
    N_i_arr = [100, 250, 500, 750, 1000, 1250, 1500]

    rand_acc_matrix = []
    best_mean_acc = 100.0
    for i in range(3):   
        rand_acc = []
        w_std = np.random.rand()*100
        b_std = np.random.rand()*100
        print(w_std, b_std)
        kernel_fn = get_kernel(3, w_std, b_std)
        for N_i in tqdm(N_i_arr):
            prediction = KernelFlowsPJAX.kernel_regression(kernel_fn, X_train=X_train[:N_i], 
                                                            Y_train=Y_train[:N_i], X_test=X_test)
            prediction_labels = np.argmax(prediction, axis=1)
            accurary = accuracy_score(prediction_labels, Y_test) * 100
            rand_acc.append(accurary)
        if best_mean_acc > np.mean(rand_acc):
            best_mean_acc = np.mean(rand_acc)
            sigma_w = w_std
            sigma_b = b_std
            
        rand_acc_matrix.append(np.array(rand_acc))

    rand_acc_matrix = np.array(rand_acc_matrix)
    rand_min = rand_acc_matrix.min(axis=0)
    rand_max = rand_acc_matrix.max(axis=0)

    KF_JAX = KernelFlowsPJAX(kernel_layers=3, kernel_output_dim=10)
    rho_values = KF_JAX.fit_finite_difference(X=X_train, Y=Y_train, batch_size=600, iterations=1000,
                                            init_sigma_w=sigma_w, init_sigma_b=sigma_b)
    fig, ax = plt.subplots(1,1)
    ax.plot(rho_values)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("$\\rho$")
    m, b = np.polyfit(np.arange(0, len(rho_values)), rho_values, 1)
    ax.plot(np.arange(0, len(rho_values)), m*np.arange(0, len(rho_values)) + b, label=f"""Best fit line: y = {m:.6f}x + {b:.2f}""")
    ax.set_ylim((0, 1))
    plt.legend()
    fig.savefig('figs/nngp_parametric_kf_finitediff_convergence.png')

    trained_acc = []
    for N_i in tqdm(N_i_arr):
        prediction = KernelFlowsPJAX.kernel_regression(KF_JAX.optimized_kernel, X_train=X_train[:N_i], Y_train=Y_train[:N_i], X_test=X_test)
        prediction_labels = np.argmax(prediction, axis=1)
        trained_acc.append(accuracy_score(prediction_labels, Y_test) * 100)

    fig, ax = plt.subplots(1,1)
    ax.plot(N_i_arr, trained_acc, 'o-', label='Finite Difference Trained NNGP')
    ax.fill_between(N_i_arr, rand_min, rand_max, alpha=0.25, label='NNGPs with randomly initialized $\sigma_w$ and $\sigma_b$')
    ax.set_ylim((0, 100))
    ax.set_xlabel("Number of input samples used for Kernel Regression $N_I$")
    ax.set_ylabel("Accuracy")
    plt.yticks(np.arange(0, 101, 5.0))
    plt.legend(loc="lower left", prop={'size': 6})
    fig.savefig('figs/nngp_parametric_kf_finitediff_accuracy.png')
