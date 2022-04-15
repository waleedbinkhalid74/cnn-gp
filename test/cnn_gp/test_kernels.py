from re import S
from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import os
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import batch_creation
import pytest
import numpy as np

def test_kernel_evaluation():
    """Sanity check to ensure if reproducability exists upon changes made to source code.
    """
    model = Sequential(Conv2d(kernel_size=3),
                        ReLU(),
                        Conv2d(kernel_size=3, stride=2),
                        ReLU(),
                        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
                    )

    X_test = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_input.pt")
    K_xx = model(X_test, X_test)
    K_xx_compare = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_output.pt")
    assert torch.equal(K_xx, K_xx_compare)


def test_batch_sample_creation():
    """Check if lengths of batch and samples match
    """
    X = torch.rand((100, 1, 2,2))
    batch_size = 50
    sample_proportion = 0.5
    samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=100,
                                                       batch_size=batch_size,
                                                       sample_proportion=sample_proportion)

    samples1, batches1 = batch_creation(X, batch_size=50, sample_proportion=0.5)

    assert len(batches) == batch_size
    assert len(samples) == batch_size * sample_proportion
    assert len(samples1) == len(samples)
    assert len(batches) == len(batches1)

    X_batch = X[batches]
    X_sample = X_batch[samples]

    assert X_batch.shape == (50,1,2,2)
    assert X_sample.shape == (25,1,2,2)

    X_batch1 = X[batches1]
    X_sample1 = X_batch1[samples1]

    assert X_batch1.shape == (50,1,2,2)
    assert X_sample1.shape == (25,1,2,2)

def test_batch_creation_error_handling():
    """Check if Exceptions are caught
    """
    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=12,
                                                        sample_proportion=0.5)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=3,
                                                        sample_proportion=1.1)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=10,
                                                        batch_size=3,
                                                        sample_proportion=-0.5)


def test_pi_matrix():
    sample_indices = np.array([0,2,7,8,9])
    N_c = 5
    N_f = 10
    pi_matrix_compare = torch.Tensor([[1,0,0,0,0,0,0,0,0,0],
                                   [0,0,1,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,1,0,0],
                                   [0,0,0,0,0,0,0,0,1,0],
                                   [0,0,0,0,0,0,0,0,0,1]])
    pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices,dimension=(N_c, N_f))
    print(pi_matrix_compare)
    print(pi_matrix)
    assert torch.equal(pi_matrix, pi_matrix_compare)
