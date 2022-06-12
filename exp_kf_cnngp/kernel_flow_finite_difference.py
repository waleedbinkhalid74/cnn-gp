import copy
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

from utils import get_dataset, get_label_from_probability
from KernelFlow import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(_):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=FLAGS.dataset, train_size=50000, val_size=1000, device=DEVICE)

    N_i_arr = np.arange(100, 1600, 100)

    rand_acc_matrix = []
    # Getting accuracy for randomly initialized CNNGP
    if FLAGS.CNNGP_model == 'alonso_etal_convnet': 
        rand_acc = []
        cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
        for N_i in tqdm(N_i_arr):
            Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                        Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                        regularization_lambda=0.0001, blocksize=250, 
                                                                                        device=DEVICE)

            Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
            rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)

    else:
        mean_accuracy = 100.0
        for i in range(5):
            cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
            rand_acc = []
            print(f"""Parameters: var_weight = {cnn_gp.var_weight}, var_bias = {cnn_gp.var_bias}""")
            for N_i in tqdm(N_i_arr):
                Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                            Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                            regularization_lambda=0.0001, blocksize=250, 
                                                                                            device=DEVICE)

                Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
                rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)
            if mean_accuracy > np.mean(rand_acc):
                mean_accuracy = np.mean(rand_acc)
                cnn_gp = copy.deepcopy(cnn_gp)
            rand_acc_matrix.append(np.array(rand_acc))

        rand_acc_matrix = np.array(rand_acc_matrix)
        rand_min = rand_acc_matrix.min(axis=0)
        rand_max = rand_acc_matrix.max(axis=0)

    # Optimizing via finite difference
    KF_finite_diff = KernelFlowsTorch(cnn_gp, device=DEVICE, regularization_lambda=1e-4)
    start = time.time()
    KF_finite_diff.fit(X_train, Y_train, iterations=500, batch_size=600,
                            sample_proportion=0.5, method='finite difference')
    stop = time.time()
    print(f"""Finite Difference took {stop - start} seconds to fit for 50 iterations. One iteration took on average {(stop - start) / 50} seconds""")

    fig, ax = plt.subplots(1,1)
    ax.plot(KF_finite_diff.rho_values)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("$\\rho$")
    m, b = np.polyfit(np.arange(0, len(KF_finite_diff.rho_values)), KF_finite_diff.rho_values, 1)
    ax.plot(np.arange(0, len(KF_finite_diff.rho_values)), m*np.arange(0, len(KF_finite_diff.rho_values)) + b, label=f"""Best fit line: y = {m:.6f}x + {b:.2f}""")
    ax.set_ylim((0, 1))
    plt.legend()
    fig.savefig('./figs/finite_diff_rho_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')
    # plt.show()
    trained_acc = []
    for N_i in tqdm(N_i_arr):
        Y_predictions_trained = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                regularization_lambda=0.0001, blocksize=250, 
                                                                                device=DEVICE)
        Y_predictions_trained_labels = np.argmax(Y_predictions_trained.cpu(), axis=1)
        trained_acc.append(accuracy_score(Y_predictions_trained_labels, Y_test.cpu().numpy()) * 100)

    print(f"""Final Parameters: var_weight = {cnn_gp.var_weight}, var_bias = {cnn_gp.var_bias}""")

    fig, ax = plt.subplots(1,1)
    if FLAGS.CNNGP_model == 'alonso_etal_convnet':
        ax.plot(N_i_arr, rand_acc, '-*', label='ConvNet-GP with parameters from Garriga-Alonso et al')
    else:
        # ax.plot(N_i_arr, rand_acc, '-*', label='CNNGP with randomly initialized $\sigma_w$ and $\sigma_b$')
        ax.fill_between(N_i_arr, rand_min, rand_max, alpha=0.25, label='CNNGPs with randomly initialized $\sigma_w$ and $\sigma_b$')

    ax.plot(N_i_arr, trained_acc, '-o', label='Finite Difference Trained CNNGP')

    # If experiment is done with convnet then the convnet model in Alonso et al is also compared
    if FLAGS.CNNGP_model == 'convnet':
        convnet_alonso_etal = kernel_flow_configs.get_CNNGP('alonso_etal_convnet', device=DEVICE)
        convnet_alonso_etal_acc = []
        for N_i in tqdm(N_i_arr):
            Y_predictions_trained = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                    Y_train=Y_train[:N_i], kernel=convnet_alonso_etal, 
                                                                                    regularization_lambda=0.0001, blocksize=250,
                                                                                    device=DEVICE)
            Y_predictions_trained_labels = np.argmax(Y_predictions_trained.cpu(), axis=1)
            convnet_alonso_etal_acc.append(accuracy_score(Y_predictions_trained_labels, Y_test.cpu().numpy()) * 100)
        ax.plot(N_i_arr, convnet_alonso_etal_acc, '-v', label='ConvNet-GP with parameters from Garriga-Alonso et al')

    ax.set_xlabel("Number of input samples used for Kernel Regression $N_I$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,100))
    
    plt.legend()
    plt.show()
    fig.savefig('./figs/finite_difference_accuracy_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "convnet",
                    "which CNNGP model to test on. For random convnet use convnet. For Convnet from Alonso et al use alonso_etal_convnet. For simple model use simple.")
    f.DEFINE_string("dataset", "mnist",
                "which dataset to work with")
    absl.app.run(main)
