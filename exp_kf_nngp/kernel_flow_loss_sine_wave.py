import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP
from exp_kf_nngp.helpers import get_network, get_training_test_points


def get_loss_from_nn(training_dataset_size_arr):
    key = random.PRNGKey(10)
    key, x_key, y_key = random.split(key, 3)
    
    depth_vs_test_loss = {}
    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))
    low_x = 0.0
    high_x = 2*np.pi
    for training_size in tqdm(training_dataset_size_arr):
        train, test = get_training_test_points(n_train=training_size, n_test=1000, 
                                            rand_key=y_key, low_x=low_x, high_x=high_x)
        train_xs, train_ys = train
        test_xs, test_ys = test

        for n in tqdm(range(1, 6, 1)):
            network_depth = n
            init_fn, apply_fn, kernel_fn = get_network(network_depth)
            # key, net_key = random.split(key)
            _, params = init_fn(key, (-1, 1))
            learning_rate = 0.001
            training_steps = 10000

            opt_init, opt_update, get_params = optimizers.adam(learning_rate)
            opt_update = jit(opt_update)

            opt_state = opt_init(params)
            for i in range(training_steps):
                opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

            # print(loss(get_params(opt_state), *test))
            if network_depth in depth_vs_test_loss:
                depth_vs_test_loss[network_depth].append(loss(get_params(opt_state), *test))
            else:
                depth_vs_test_loss[network_depth] = [loss(get_params(opt_state), *test)]
            
    return depth_vs_test_loss

def get_loss_from_kf_rbf(training_dataset_size_arr):
    batch_size_vs_loss = {}
    low_x = 0.0
    high_x = 2*np.pi
    key = random.PRNGKey(10)
    key, x_key, y_key = random.split(key, 3)
    for training_size in tqdm(training_dataset_size_arr):
        train, test = get_training_test_points(n_train=training_size, 
                                              n_test=1000, rand_key=y_key, 
                                              low_x=low_x, high_x=high_x)
        train_xs, train_ys = train
        test_xs, test_ys = test

        mu = np.array([2.0])
        kernel_name = "RBF"
        KF_rbf = KernelFlowsNP(kernel_name, mu)
        iterations = 1000
        batch_size = training_size
        X_train_perturbed_rbf = KF_rbf.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01, reg=0.0001)
        predict_kf_rbf = KF_rbf.predict(test_xs, regu=0.0001)
        batch_size_vs_loss[str(training_size)] = 0.5 * np.mean((predict_kf_rbf - test_ys) ** 2)
    return batch_size_vs_loss

if __name__ == "__main__":
    training_dataset_size_arr = np.arange(10, 60, 10)
    depth_vs_test_loss_nn = get_loss_from_nn(training_dataset_size_arr)
    batch_size_vs_loss_kf_rbf = get_loss_from_kf_rbf(training_dataset_size_arr)


    # Plotting
    fig, ax = plt.subplots(1,1)
    for key, value in depth_vs_test_loss_nn.items():
        ax.plot(training_dataset_size_arr, depth_vs_test_loss_nn[key], 'o-', label=f""" NN Depth {str(key)}""")

    ax.plot(training_dataset_size_arr, batch_size_vs_loss_kf_rbf.values(), '*--', label='Kernel Flow - RBF')
    ax.set_xlabel("Training data size")
    ax.set_ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig("figs/sine_wave_training_mse_loss.png")