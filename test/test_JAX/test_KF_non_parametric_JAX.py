from KernelFlow import KernelFlowsNPJAX
from KernelFlow import batch_creation
import jax.numpy as jnp
import numpy as np
import torch
import jax.numpy as jnp
from jax import random
from neural_tangents import stax
from jax import jit

def get_network_sigmoid(n=2):
    var_weight = 1.5
    var_bias = 0.05
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        stax.Dense(32, W_std=var_weight, b_std=var_bias, parameterization='standard'), 
        stax.Sigmoid_like()
        ]

    init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=var_weight, b_std=var_bias, parameterization='standard')
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return init_fn, apply_fn, kernel_fn

def test_batch_sample_creation():
    """Check if lengths of batch and samples match
    """
    X = torch.rand((100, 1, 2,2))

    batch_size = 50
    sample_proportion = 0.5
    samples, batches = KernelFlowsNPJAX.batch_creation(dataset_size=100,
                                                       batch_size=batch_size,
                                                       sample_proportion=sample_proportion)

    samples1, batches1 = batch_creation(X, batch_size=50, sample_proportion=0.5)

    assert len(batches) == batch_size
    assert len(samples) == batch_size * sample_proportion
    assert len(samples1) == len(samples)
    assert len(batches) == len(batches1)

    X_batch = X[np.array(batches)]
    X_sample = X_batch[np.array(samples)]

    assert X_batch.shape == (50,1,2,2)
    assert X_sample.shape == (25,1,2,2)

    X_batch1 = X[batches1]
    X_sample1 = X_batch1[samples1]

    assert X_batch1.shape == (50,1,2,2)
    assert X_sample1.shape == (25,1,2,2)


def test_pi_matrix():
    sample_indices = jnp.array([0,2,7,8,9])
    N_c = 5
    N_f = 10
    pi_matrix_compare = jnp.array([[1,0,0,0,0,0,0,0,0,0],
                                   [0,0,1,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,1,0,0],
                                   [0,0,0,0,0,0,0,0,1,0],
                                   [0,0,0,0,0,0,0,0,0,1]])

    pi_matrix = KernelFlowsNPJAX.pi_matrix(sample_indices=sample_indices,dimension=(N_c, N_f))
    assert jnp.array_equal(pi_matrix_compare, pi_matrix)

def test_prediction():
    
    key = random.PRNGKey(10)
    train_points = 10
    test_points = 100
    noise_scale = 1e-2
    low_x = 0.0
    high_x = 2*jnp.pi

    target_fn = lambda x: jnp.sin(x)
    train_xs = jnp.linspace(low_x, high_x, train_points)
    train_xs = jnp.reshape(train_xs, (train_points, 1))

    train_ys = target_fn(train_xs)
    train = (train_xs, train_ys)
    test_xs = jnp.linspace(low_x, high_x, test_points)
    test_xs = jnp.reshape(test_xs, (test_points, 1))
    test_ys = target_fn(test_xs)
    test = (test_xs, test_ys)

    init_fn, apply_fn, kernel_fn = get_network_sigmoid(1)
    # Checking results from kernel Flow prediction
    KF_JAX = KernelFlowsNPJAX(kernel_fn, regularization_lambda=0.00001)
    iterations = 1
    batch_size = train_points
    _ = KF_JAX.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01)
    prediction_KF_JAX = KF_JAX.predict(test_xs)

    # Results from Kernel regression
    k_xx = kernel_fn(train_xs, train_xs, 'nngp')
    t_xx = kernel_fn(test_xs, train_xs, 'nngp')
    coeff, _, _, _ = jnp.linalg.lstsq(k_xx + 0.00001*jnp.eye(k_xx.shape[0]), train_ys, rcond=1e-6)
    pred = jnp.matmul(t_xx, coeff)

    # Assert both predictions are equal (or very close)
    assert jnp.all(jnp.isclose(prediction_KF_JAX, pred))
