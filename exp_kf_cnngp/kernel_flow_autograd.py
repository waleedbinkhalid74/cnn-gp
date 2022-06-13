import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')

from utils import get_dataset, get_label_from_probability
from KernelFlow import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(_):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=FLAGS.dataset, train_size=50000, val_size=1000, device=DEVICE)
    cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
    N_i_arr = np.arange(100, 1600, 100)
    rand_acc = []

    print(f"""Initial Parameters: var_weight = {cnn_gp.var_weight}, \n var_bias = {cnn_gp.var_bias}""")



    # Getting accuracy for randomly initialized CNNGP
    for N_i in tqdm(N_i_arr):
        Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                    Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                    regularization_lambda=0.0001, blocksize=250, 
                                                                                    device=DEVICE)

        Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
        rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)

    # plt.plot(rand_acc)
    # plt.ylim((0,100))
    # plt.show()

    KF_autograd = KernelFlowsTorch(cnn_gp, device=DEVICE, regularization_lambda=1e-4)
    iteration_count = 1000
    start = time.time()
    KF_autograd.fit(X_train, Y_train, iterations=iteration_count, batch_size=600,
                            sample_proportion=0.5, method='autograd')
    stop = time.time()
    print(f"""Automatic Differentiation took {stop - start} seconds to fit for {iteration_count} iterations. One iteration took on average {(stop - start) / iteration_count} seconds""")

    fig, ax = plt.subplots(1,1)
    ax.plot(KF_autograd.rho_values)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("$\\rho$")
    m, b = np.polyfit(np.arange(0, len(KF_autograd.rho_values)), KF_autograd.rho_values, 1)
    ax.plot(np.arange(0, len(KF_autograd.rho_values)), m*np.arange(0, len(KF_autograd.rho_values)) + b, label=f"""Best fit line: y = {m:.6f}x + {b:.2f}""")
    ax.set_ylim((0, 1))
    plt.legend()
    plt.show()
    # fig.savefig('./figs/autograd_rho_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

    print(f"""Final Parameters: var_weight = {cnn_gp.var_weight}, var_bias = {cnn_gp.var_bias}""")

    trained_acc = []
    for N_i in tqdm(N_i_arr):
        Y_predictions_trained = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                regularization_lambda=0.0001, blocksize=250, 
                                                                                device=DEVICE)
        Y_predictions_trained_labels = np.argmax(Y_predictions_trained.cpu(), axis=1)
        trained_acc.append(accuracy_score(Y_predictions_trained_labels, Y_test.cpu().numpy()) * 100)



    fig, ax = plt.subplots(1,1)
    ax.plot(N_i_arr, rand_acc, '-*', label='CNNGP with randomly initialized $\sigma_w$ and $\sigma_b$')
    ax.plot(N_i_arr, trained_acc, '-o', label='Autograd Trained CNNGP')
    ax.set_xlabel("Number of input samples used for Kernel Regression $N_I$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,100))
    plt.yticks(np.arange(0, 101, 5.0))
    plt.legend()
    plt.show()
    # fig.savefig('./figs/autograd_accuracy_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "simple",
                    "which CNNGP model to test on")
    f.DEFINE_string("dataset", "mnist",
                "which dataset to work with")
    absl.app.run(main)
