from KernelFlow import KernelFlowsNPJAX
from KernelFlow import batch_creation
import numpy as np
import torch
import jax.numpy as jnp


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