import KernelFlow
from KernelFlow import KernelFlowsNPAutograd
from KernelFlow import KernelFlowsNP
from KernelFlow import KernelFlowsNPJAX
import numpy as np
from neural_tangents import stax
from jax import jit
import jax.numpy as jnp

if __name__ == "__main__":
    np.random.seed(1)
    init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(10, W_std=1.5, b_std=0.05), stax.Relu(),
                stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    kf_frechet = KernelFlowsNP('RBF', [2.0])
    kf_autograd = KernelFlowsNPAutograd('RBF', [2.0])
    kf_jax = KernelFlowsNPJAX(kernel_fn, regularization_lambda=0.000001)

    X = np.random.rand(1000,1)
    Y = np.random.rand(1000,1)
    # kf_frechet.fit(X, Y, 2, 100)
    # kf_autograd.fit(X, Y, 2, 100)
    kf_jax.fit(X=X, Y=Y, iterations=2, batch_size=100)

