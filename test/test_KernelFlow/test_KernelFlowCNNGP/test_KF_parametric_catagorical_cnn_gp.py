from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
from KernelFlow import KernelFlowsTorch
import numpy as np
from KernelFlow import batch_creation
import pytest
import torch.nn.functional as F

def test_batch_sample_creation():
    """Check if lengths of batch and samples match
    """
    X = torch.rand((100, 1, 2,2))
    batch_size = 50
    sample_proportion = 0.5
    samples, batches = KernelFlowsTorch.batch_creation(dataset_size=100,
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
        samples, batches = KernelFlowsTorch.batch_creation(dataset_size=10,
                                                        batch_size=12,
                                                        sample_proportion=0.5)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsTorch.batch_creation(dataset_size=10,
                                                        batch_size=3,
                                                        sample_proportion=1.1)

    with pytest.raises(Exception):
        samples, batches = KernelFlowsTorch.batch_creation(dataset_size=10,
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
    pi_matrix = KernelFlowsTorch.pi_matrix(sample_indices=sample_indices,dimension=(N_c, N_f))
    print(pi_matrix_compare)
    print(pi_matrix)
    assert torch.equal(pi_matrix, pi_matrix_compare)

def test_rho():
    """Test if calculation of rho is correct compared to Darcy's implementation
    """
    model = Sequential(2.0, 1.0,
                Conv2d(kernel_size=3, padding=0),
                ReLU(),
                )
    K = KernelFlowsTorch(cnn_gp_kernel=model)
    X = torch.ones((10, 1, 3,3), dtype=torch.float32)
    for i in range(X.shape[0]):
        X[i] = X[i] * i

    Y = torch.arange(0,10)
    Y = F.one_hot(Y, 10)
    Y = Y.to(torch.float32)

    samples = np.array([2,3,4])
    batches = np.array([1, 3, 4, 5, 7])


    N_f = len(batches)
    N_c = len(samples)
    X_batch = X[batches]
    X_sample = X[samples]
    Y_batch = Y[batches]
    Y_sample = Y[samples]

    pi_matrix = KernelFlowsTorch.pi_matrix(sample_indices=samples, dimension=(N_c, N_f))
    rho = K.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)

    ##### TEST COMPARISION #####
    pi_comp =  np.array([[0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]], dtype=float)

    K_xx = model(X_batch, X_batch)
    K_xx = K_xx.detach().numpy()
    sample_matrix = np.matmul(pi_comp, np.matmul(K_xx, np.transpose(pi_comp)))
    inverse_data = np.linalg.inv(K_xx + 0.000001 * np.identity(K_xx.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + 0.000001 * np.identity(sample_matrix.shape[0]))
    top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
    bottom = np.matmul(Y_batch.T, np.matmul(inverse_data, Y_batch))
    rho_comp = 1 - np.trace(top)/np.trace(bottom)
    # TODO: Check if a relative tolerance of just 0.01 is acceptable.
    # Pytorch and numpy result in slightly different results due to numerical reasons
    # Is this acceptable?
    assert np.isclose(rho.detach().numpy(), rho_comp, 1e-3)

def test_blocked_kernel_eval_square_result():
    """Test if blocked kernel evaluation results in the same result as a complete evaluation
    """
    X = torch.rand((100, 1, 28, 28))
    Y = torch.rand((100, 10))

    model_untrained = Sequential(2.0, 1.0,
        Conv2d(kernel_size=3),
        ReLU(),
        Conv2d(kernel_size=3, stride=2),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
        )

    # Complete Kernel
    with torch.no_grad():
        k_full = model_untrained(X, X).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X,
                                                        Y=X,
                                                        blocksize=25,
                                                        kernel=model_untrained).numpy()

    assert np.all(np.equal(k_full, k_blocked))

def test_blocked_kernel_eval_rec_result():
    """Test if blocked kernel evaluation with remainder answer results in the same result as a complete evaluation
    """
    X = torch.rand((100, 1, 28, 28))
    Y = torch.rand((100, 10))

    model_untrained = Sequential(3.0, 5.0,
        Conv2d(kernel_size=3),
        ReLU(),
        Conv2d(kernel_size=3, stride=2),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
        )

    with torch.no_grad():

        # Complete Kernel
        k_full = model_untrained(X[:25], X[:25]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X[:25],
                                                        Y=X[:25],
                                                        blocksize=50,
                                                        kernel=model_untrained).numpy()
        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:20], X[:25]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X[:20],
                                                        Y=X[:25],
                                                        blocksize=50,
                                                        kernel=model_untrained).numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:16], X[:10]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X[:16],
                                                        Y=X[:10],
                                                        blocksize=50,
                                                        kernel=model_untrained).numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:25], X[:25]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X[:25],
                                                        Y=X[:25],
                                                        blocksize=100,
                                                        kernel=model_untrained).numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:100], X[:10]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsTorch._block_kernel_eval(X=X[:100],
                                                        Y=X[:10],
                                                        blocksize=100,
                                                        kernel=model_untrained).numpy()

        assert np.all(np.equal(k_full, k_blocked))