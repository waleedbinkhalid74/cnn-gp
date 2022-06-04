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
from KernelFlow import KernelFlowsNPJAX

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

if __name__ == "__main__":
    key = random.PRNGKey(10)
    train_points = 10
    test_points = 100
    noise_scale = 1e-2
    target_fn = lambda x: np.sin(x)
    key, x_key, y_key = random.split(key, 3)
    train_xs = random.uniform(x_key, (train_points, 1), minval=-np.pi, maxval=np.pi)
    train_ys = target_fn(train_xs)
    train_ys += noise_scale * random.normal(y_key, (train_points, 1))
    train = (train_xs, train_ys)
    test_xs = np.linspace(-np.pi, np.pi, test_points)
    test_xs = np.reshape(test_xs, (test_points, 1))
    test_ys = target_fn(test_xs)
    test = (test_xs, test_ys)

    fig, ax = plt.subplots(1, 1)

    for n in tqdm(range(0, 10, 3)):
        init_fn, apply_fn, kernel_fn = get_network(n*3 + 1)
        key, net_key = random.split(key)
        _, params = init_fn(net_key, (-1, 1))

        learning_rate = 0.001
        training_steps = 10000

        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_update = jit(opt_update)
        loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
        grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

        train_losses = []
        test_losses = []

        opt_state = opt_init(params)
        for i in range(training_steps):
            opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

            train_losses += [loss(get_params(opt_state), *train)]
            test_losses += [loss(get_params(opt_state), *test)]
        ax.plot(test_xs, apply_fn(get_params(opt_state), test_xs), linewidth=2, label=f"""Depth = {n*3 + 1}""", alpha=0.5)

    init_fn, apply_fn, kernel_fn = get_network(1)

    # Evaluation with Kernel Flow with NNGP Kernel 
    KF_JAX = KernelFlowsNPJAX(kernel_fn)
    iterations = 500
    batch_size = 10
    _ = KF_JAX.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01)
    prediction_KF_JAX = KF_JAX.predict(test_xs)

    # Evaluation with Kernel Flow with RBF Kernel
    mu = np.array([2.0])
    kernel_name = "RBF"
    KF_RBF = KernelFlowsNP(kernel_name, mu)
    X_train_perturbed_rbf = KF_RBF.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01)
    predict_kf_rbf = KF_RBF.predict(test_xs)

    ax.plot(test_xs, prediction_KF_JAX, '--', linewidth=2, label='Kernel Flow - NNGP')
    ax.plot(test_xs, predict_kf_rbf, '--', linewidth=2, label='Kernel Flow - RBF')
    ax.plot(train_xs, train_ys, 'o', label="Train")
    ax.plot(test_xs, test_ys, '--', linewidth=2, label='Test')
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    plt.legend()
    fig.savefig('figs/dummy.png')