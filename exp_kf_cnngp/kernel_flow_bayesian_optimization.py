import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os
from skopt.plots import plot_convergence
    
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

    # Getting accuracy for randomly initialized CNNGP
    for N_i in tqdm(N_i_arr):
        Y_predictions_rand_cnngp = KernelFlowsTorch.kernel_regression(X_test=X_test, X_train=X_train[:N_i], 
                                                                                    Y_train=Y_train[:N_i], kernel=cnn_gp, 
                                                                                    regularization_lambda=0.0001, blocksize=250, 
                                                                                    device=DEVICE)

        Y_predictions_rand_cnngp_labels = get_label_from_probability(Y_predictions_rand_cnngp)
        rand_acc.append(accuracy_score(Y_predictions_rand_cnngp_labels, Y_test.cpu().numpy()) * 100)
        
    # Training with Bayesian Optimization
    parameter_bounds = [(1e-3, 100.0), (0.0, 100.0)]
    KF_BO = KernelFlowsTorch(cnn_gp, device=DEVICE, regularization_lambda=1e-4)
    res = KF_BO.fit(X_train, Y_train, iterations=50, batch_size=1200, 
                    sample_proportion=0.5, parameter_bounds_BO=parameter_bounds, 
                    random_starts=15, method='bayesian optimization')

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
    ax.plot(N_i_arr, rand_acc, '-*', label='CNNGP with randomly initialized $\sigma_w$ and $\sigma_b$')
    ax.plot(N_i_arr, bo_acc, '-o', label='Bayesian Optimization Trained CNNGP')
    ax.set_xlabel("Number of input samples used for Kernel Regression $N_I$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0,100))
    plt.legend()
    plt.show()
    fig.savefig('./figs/bayesian_optimization_accuracy_' + FLAGS.CNNGP_model + "_" + FLAGS.dataset + '.png')

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "covnet",
                    "which CNNGP model to test on")
    f.DEFINE_string("dataset", "mnist",
                "which dataset to work with")
    absl.app.run(main)
