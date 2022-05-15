from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
from KernelFlow import KernelFlowsCNNGP
import numpy as np
from KernelFlow import batch_creation
import pytest
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    pi = torch.zeros((N_c, N_f)).to(DEVICE)
    pi_matrix = KernelFlowsCNNGP._pi_matrix(pi_matrix=pi, sample_indices=sample_indices,dimension=(N_c, N_f))
    assert torch.equal(pi_matrix.cpu(), pi_matrix_compare.cpu())

def test_rho():
    """Test if calculation of rho is correct compared to Darcy's implementation
    """
    model = Sequential(
                Conv2d(kernel_size=3, padding=0),
                ReLU(),
                )
    model.to(DEVICE)
    K = KernelFlowsCNNGP(cnn_gp_kernel=model)
    X = torch.rand((10, 1, 3,3), dtype=torch.float32).to(DEVICE)
    for i in range(X.shape[0]):
        X[i] = X[i] * i

    Y = torch.arange(0,10)
    Y = F.one_hot(Y, 10)
    Y = Y.to(torch.float32)
    Y = Y.to(DEVICE)

    samples = np.array([2,3,4])
    batches = np.array([1, 3, 4, 5, 7])

    N_f = len(batches)
    N_c = len(samples)
    X_batch = X[batches]
    X_sample = X[samples]
    Y_batch = Y[batches]
    Y_sample = Y[samples]

    pi = torch.zeros((N_c, N_f)).to(DEVICE)
    pi_matrix = KernelFlowsCNNGP._pi_matrix(pi_matrix=pi, sample_indices=samples, dimension=(N_c, N_f))
    rho = K.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)

    ##### TEST COMPARISION #####
    pi_comp =  np.array([[0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]], dtype=float)

    assert np.all(np.equal(pi_matrix.cpu().numpy(), pi_comp))

    K_xx = model(X_batch, X_batch)
    K_xx = K_xx.cpu().detach().numpy()
    sample_matrix = np.matmul(pi_comp, np.matmul(K_xx, np.transpose(pi_comp)))
    inverse_data = np.linalg.inv(K_xx + 0.000001 * np.identity(K_xx.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + 0.000001 * np.identity(sample_matrix.shape[0]))
    top = np.matmul(Y_sample.cpu().T, np.matmul(inverse_sample, Y_sample.cpu()))
    bottom = np.matmul(Y_batch.T.cpu(), np.matmul(inverse_data, Y_batch.cpu()))
    rho_comp = 1 - np.trace(top)/np.trace(bottom)
    # TODO: Check if a relative tolerance of just 0.01 is acceptable.
    # Pytorch and numpy result in slightly different results due to numerical reasons
    # Is this acceptable?
    print(rho.cpu().detach().numpy(), rho_comp)
    assert np.isclose(rho.cpu().detach().numpy(), rho_comp, 1e-3)

def test_blocked_kernel_eval_square_result():
    """Test if blocked kernel evaluation results in the same result as a complete evaluation
    """
    X = torch.rand((100, 1, 28, 28))
    Y = torch.rand((100, 10))

    model_untrained = Sequential(
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
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X,
                                                        Y=X,
                                                        blocksize=25,
                                                        kernel=model_untrained).cpu().numpy()

    assert np.all(np.equal(k_full, k_blocked))

def test_blocked_kernel_eval_rec_result():
    """Test if blocked kernel evaluation with remainder answer results in the same result as a complete evaluation
    """
    X = torch.rand((5000, 1, 28, 28))
    Y = torch.rand((5000, 10))

    model_untrained = Sequential(
        Conv2d(kernel_size=3),
        ReLU(),
        Conv2d(kernel_size=3, stride=2),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
        )

    with torch.no_grad():

        # Complete Kernel
        k_full = model_untrained(X[:125], X[:125]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X[:125],
                                                        Y=X[:125],
                                                        blocksize=50,
                                                        kernel=model_untrained).cpu().numpy()
        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:120], X[:125]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X[:120],
                                                        Y=X[:125],
                                                        blocksize=50,
                                                        kernel=model_untrained).cpu().numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:116], X[:110]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X[:116],
                                                        Y=X[:110],
                                                        blocksize=50,
                                                        kernel=model_untrained).cpu().numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:125], X[:125]).detach().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X[:125],
                                                        Y=X[:125],
                                                        blocksize=100,
                                                        kernel=model_untrained).cpu().numpy()

        assert np.all(np.equal(k_full, k_blocked))

        # Complete Kernel
        k_full = model_untrained(X[:1000], X[:100]).detach().cpu().numpy()

        # Blockwise Kernel
        k_blocked = KernelFlowsCNNGP._block_kernel_eval(X=X[:1000],
                                                        Y=X[:100],
                                                        blocksize=100,
                                                        kernel=model_untrained).cpu().numpy()

        assert np.all(np.equal(k_full, k_blocked))