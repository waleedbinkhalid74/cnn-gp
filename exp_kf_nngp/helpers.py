import jax.numpy as np
from jax import jit
from neural_tangents import stax
from jax import random

def get_network(n=1):
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        stax.Dense(32, W_std=1.5, b_std=0.05), 
        stax.Relu()
        ]

    init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return init_fn, apply_fn, kernel_fn



def get_training_test_points(n_train, n_test, rand_key, low_x = 0.0, high_x = 2*np.pi):
    train_points = n_train
    test_points = n_test
    noise_scale = 0*1e-2

    target_fn = lambda x: np.sin(x)

    # train_xs = random.uniform(x_key, (train_points, 1), minval=low_x, maxval=high_x)
    train_xs = np.linspace(low_x, high_x, train_points)
    train_xs = np.reshape(train_xs, (train_points, 1))

    train_ys = target_fn(train_xs)

    # key, x_key, y_key = random.split(key, 3)
    train_ys += noise_scale * random.normal(rand_key, (train_points, 1))
    train = (train_xs, train_ys)
    test_xs = np.linspace(low_x, high_x, test_points)
    test_xs = np.reshape(test_xs, (test_points, 1))

    test_ys = target_fn(test_xs)
    test = (test_xs, test_ys)
    return train, test