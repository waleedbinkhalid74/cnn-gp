import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os
from skopt.plots import plot_convergence
import time
sys.path.insert(0, os.getcwd() + '/.')

from utils import get_dataset, get_label_from_probability
from KernelFlow.Torch.KF_parametric_catagorical_torch import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(_):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=FLAGS.dataset, train_size=50000, val_size=1000, device=DEVICE)

    N_i_arr = np.arange(100, 200, 100)

    rand_acc_matrix = []
    # Getting accuracy for randomly initialized CNNGP
    if FLAGS.CNNGP_model == 'alonso_etal_convnet': 
        rand_acc = []
        for N_i in tqdm(N_i_arr):
            Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                        Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                        regularization_lambda=0.0001, blocksize=250, 
                                                                                        device=DEVICE)

            Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
            rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)

    else:
        mean_accuracy = 100.0
        for i in range(1):
            cnn_gp_temp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)
            rand_acc = []
            for N_i in tqdm(N_i_arr):
                Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                            Y_train=Y_train[:N_i], kernel=cnn_gp_temp, 
                                                                                            regularization_lambda=0.0001, blocksize=250, 
                                                                                            device=DEVICE)

                Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
                rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)
            if mean_accuracy > np.mean(rand_acc):
                mean_accuracy = np.mean(rand_acc)
                cnn_gp = copy.deepcopy(cnn_gp_temp)
            rand_acc_matrix.append(np.array(rand_acc))

        rand_acc_matrix = np.array(rand_acc_matrix)
        rand_min = rand_acc_matrix.min(axis=0)
        rand_max = rand_acc_matrix.max(axis=0)
        plt.fill_between(N_i_arr, rand_min, rand_max, alpha=0.25, label='CNNGPs with randomly initialized $\sigma_w$ and $\sigma_b$')
        plt.show()
    # Training with Bayesian Optimization
    parameter_bounds = [(1e-3, 200.0), (0.0, 200.0)]
    KF_BO = KernelFlowsTorch(cnn_gp, device=DEVICE, regularization_lambda=1e-4)
    iteration_count = 50
    start = time.time()
    res = KF_BO.fit(X_train, Y_train, iterations=iteration_count, batch_size=1200, 
                    sample_proportion=0.5, parameter_bounds_BO=parameter_bounds, 
                    random_starts=iteration_count // 2, method='bayesian optimization')
    stop = time.time()
    print(f"""Bayesian Optimization took {stop - start} seconds to fit for {iteration_count} iterations. One iteration took on average {(stop - start) / iteration_count} seconds""")

    fig, ax = plt.subplots(1,1)
    plot_convergence(res, ax=ax)
    ax.set_ylim((0,1))
    plt.show()
    fig.savefig('./figs/bayesian_optimization_convergence_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

    bo_acc = []
    for N_i in tqdm(N_i_arr):
        Y_predictions_trained = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                regularization_lambda=0.0001, blocksize=250, 
                                                                                device=DEVICE)
        Y_predictions_trained_labels = np.argmax(Y_predictions_trained.cpu(), axis=1)
        bo_acc.append(accuracy_score(Y_predictions_trained_labels, Y_test.cpu().numpy()) * 100)

    fig, ax = plt.subplots(1,1)
    if FLAGS.CNNGP_model == 'alonso_etal_convnet':
        ax.plot(N_i_arr, rand_acc, '-*', label='ConvNet-GP with parameters from Garriga-Alonso et al')
    else:
        # ax.plot(N_i_arr, rand_acc, '-*', label='CNNGP with randomly initialized $\sigma_w$ and $\sigma_b$')
        ax.fill_between(N_i_arr, rand_min, rand_max, alpha=0.25, label='CNNGPs with randomly initialized $\sigma_w$ and $\sigma_b$')
        

    ax.plot(N_i_arr, bo_acc, '-o', label='Bayesian Optimization Trained CNNGP')

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
    plt.yticks(np.arange(0, 101, 5.0))
    plt.legend()
    plt.show()
    fig.savefig('./figs/bayesian_optimization_accuracy_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "convnet",
                    "which CNNGP model to test on. For random convnet use convnet. For convnet from Alonso et al use alonso_etal_convnet. For simple model use simple.")
    f.DEFINE_string("dataset", "mnist",
                "which dataset to work with")
    absl.app.run(main)
