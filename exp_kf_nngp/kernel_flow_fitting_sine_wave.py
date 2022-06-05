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
from exp_kf_nngp.helpers import get_network, get_training_test_points



if __name__ == "__main__":

    key = random.PRNGKey(10)
    key, x_key, y_key = random.split(key, 3)
    low_x = 0.0
    high_x = 2*np.pi
    train_points = 10
    test_points = 1000
    train, test = get_training_test_points(n_train=train_points, n_test=test_points, 
                              low_x=low_x, high_x=high_x, rand_key=y_key)
    train_xs, train_ys = train
    test_xs, test_ys = test

    fig, ax = plt.subplots(1, 1)

    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    for n in tqdm(range(1, 6, 1)):
        network_depth = n*2 + 1
        init_fn, apply_fn, kernel_fn = get_network(network_depth)
        _, params = init_fn(key, (-1, 1))

        learning_rate = 0.001
        training_steps = 10000

        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_update = jit(opt_update)

        opt_state = opt_init(params)
        for i in range(training_steps):
            opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)
            # train_losses += [loss(get_params(opt_state), *train)]
        test_losses = loss(get_params(opt_state), *test)

        ax.plot(test_xs, apply_fn(get_params(opt_state), test_xs), linewidth=2, label=f"""Depth = {network_depth}, Test loss = {test_losses:0.5f} """, alpha=0.75)

    init_fn, apply_fn, kernel_fn = get_network(1)

    # Evaluation with Kernel Flow with NNGP Kernel 
    KF_JAX = KernelFlowsNPJAX(kernel_fn, regularization_lambda=0.001)
    iterations = 1
    batch_size = train_points
    _ = KF_JAX.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01)
    prediction_KF_JAX = KF_JAX.predict(test_xs)

    # Evaluation with Kernel Flow with RBF Kernel
    mu = np.array([2.0])
    kernel_name = "RBF"
    KF_RBF = KernelFlowsNP(kernel_name, mu)
    X_train_perturbed_rbf = KF_RBF.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01, reg=0.00001)
    predict_kf_rbf = KF_RBF.predict(test_xs)

    ax.plot(test_xs, prediction_KF_JAX, '--', linewidth=2, label='Kernel Flow - NNGP')
    ax.plot(test_xs, predict_kf_rbf, '--', linewidth=2, label='Kernel Flow - RBF')
    ax.plot(train_xs, train_ys, 'o', label="Train")
    ax.plot(test_xs, test_ys, '--', linewidth=2, label='Test')
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    plt.legend()
    fig.savefig('figs/dummy.png')