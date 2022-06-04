import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad
import functools
import neural_tangents as nt
from neural_tangents import stax
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP

def format_plot(x=None, y=None):
  # plt.grid(False)
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

legend = functools.partial(plt.legend, fontsize=10)

def get_network(n=1):
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        stax.Dense(512, W_std=1.5, b_std=0.05), 
        stax.Erf()
        ]

    init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return init_fn, apply_fn, kernel_fn

def get_loss_from_nn(training_dataset_size_arr):
    key = random.PRNGKey(10)
    key, x_key, y_key = random.split(key, 3)
    
    depth_vs_test_loss = {}
    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    for training_size in tqdm(training_dataset_size_arr):
        train_points = training_size
        test_points = 1000
        noise_scale = 1e-2

        target_fn = lambda x: np.sin(x)


        train_xs = random.uniform(x_key, (train_points, 1), minval=-np.pi, maxval=np.pi)

        train_ys = target_fn(train_xs)

        train_ys += noise_scale * random.normal(y_key, (train_points, 1))
        train = (train_xs, train_ys)
        test_xs = np.linspace(-np.pi, np.pi, test_points)
        test_xs = np.reshape(test_xs, (test_points, 1))

        test_ys = target_fn(test_xs)
        test = (test_xs, test_ys)

        for n in tqdm(range(0, 13, 3)):
            network_depth = n*3 + 1
            init_fn, apply_fn, kernel_fn = get_network(network_depth)
            # key, net_key = random.split(key)
            _, params = init_fn(key, (-1, 1))
            learning_rate = 0.001
            training_steps = 10000

            opt_init, opt_update, get_params = optimizers.adam(learning_rate)
            opt_update = jit(opt_update)

            train_losses = []
            test_losses = []

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
    key = random.PRNGKey(10)
    batch_size_vs_loss = {}
    key, x_key, y_key = random.split(key, 3)
    for training_size in tqdm(training_dataset_size_arr):
        train_points = training_size
        test_points = 1000
        noise_scale = 1e-2
        target_fn = lambda x: np.sin(x)
        train_xs = random.uniform(x_key, (train_points, 1), minval=-np.pi, maxval=np.pi)

        train_ys = target_fn(train_xs)

        train_ys += noise_scale * random.normal(y_key, (train_points, 1))
        train = (train_xs, train_ys)
        test_xs = np.linspace(-np.pi, np.pi, test_points)
        test_xs = np.reshape(test_xs, (test_points, 1))

        test_ys = target_fn(test_xs)
        test = (test_xs, test_ys)

        mu = np.array([2.0])
        kernel_name = "RBF"
        KF_rbf = KernelFlowsNP(kernel_name, mu)
        iterations = 1000
        batch_size = train_points
        X_train_perturbed_rbf = KF_rbf.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01, reg=0.0001)
        predict_kf_rbf = KF_rbf.predict(test_xs)
        batch_size_vs_loss[str(training_size)] = 0.5 * np.mean(predict_kf_rbf - test_ys) ** 2
    return batch_size_vs_loss

if __name__ == "__main__":
    training_dataset_size_arr = np.arange(20, 120, 20)
    depth_vs_test_loss_nn = get_loss_from_nn(training_dataset_size_arr)
    batch_size_vs_loss_kf_rbf = get_loss_from_kf_rbf(training_dataset_size_arr)

    fig, ax = plt.subplots(1,1)
    for key, value in depth_vs_test_loss_nn.items():
        ax.plot(training_dataset_size_arr, depth_vs_test_loss_nn[key], 'o-', label=str(key))

    ax.plot(training_dataset_size_arr, batch_size_vs_loss_kf_rbf.values(), '*--', label='Kernel Flow - RBF')
    ax.set_xlabel("Training data size")
    ax.set_ylabel("MSE Loss")
    plt.legend()
    fig.savefig("figs/sine_wave_training_mse_loss.png")