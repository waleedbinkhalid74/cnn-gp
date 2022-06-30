import time
from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad , random
from tqdm import tqdm
import jax

class KernelFlowsJAXBase():
    def __init__(self) -> None:
        pass

    @staticmethod
    def sample_selection(data_size: int, size: int, key) -> np.ndarray:
        """Selects an iid sample from the dataset without replacement

        Args:
            data_size (int): size of data to sample from
            size (int): number of items to sample
            key (_type_): JAX generated random key

        Returns:
            np.array: sampled indices
        """

        indices = np.arange(data_size)
        choices = random.choice(key, indices, [size], replace=False)
        # sample_indices = np.sort(np.random.choice(indices, size, replace= False))
        sample_indices = np.sort(choices)
        return sample_indices
    
    @staticmethod
    def batch_creation(dataset_size:int, batch_size: int, sample_proportion: float = 0.5) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
        """Creates a batch N_f and sample N_c from available data for kernel regression

        Args:
            dataset_size (int): size of entire dataset from which to create batches
            batch_size (int): N_f samples from entire batch
            sample_proportion (float): N_c samples from entire batch where N_c < N_f

        Returns:
            (jax.numpy.ndarray, jax.numpy.ndarray):  sample and batch indices respectively
        """
        # Error handling
        if batch_size > dataset_size:
            raise Exception("Batch size must be lesser than dataset size.")

        if sample_proportion > 1.0 or sample_proportion < 0.0:
            raise Exception("Sample proportion should be between 0 and 1")

        key = random.PRNGKey(int(time.time()))
        key, batch_key, sample_key = random.split(key, 3)
        # batch creation
        if batch_size == False:
            batch_indices = np.arange(dataset_size)
        elif 0 < batch_size <= 1:
            batch_size = int(dataset_size * batch_size)
            batch_indices = KernelFlowsJAXBase.sample_selection(dataset_size, batch_size, batch_key)
            data_batch = len(batch_indices)
        else:
            batch_size = int(batch_size)
            batch_indices = KernelFlowsJAXBase.sample_selection(dataset_size, batch_size, batch_key)
            data_batch = len(batch_indices)

        # Sample from the mini-batch
        sample_size = int(np.ceil(data_batch*sample_proportion))
        sample_indices = KernelFlowsJAXBase.sample_selection(data_batch, sample_size, sample_key)

        return sample_indices, batch_indices

    @staticmethod
    def pi_matrix(sample_indices: np.ndarray, dimension: Tuple) -> jax.numpy.ndarray:
        """Evaluates the pi matrix. pi matrix is the corresponding Nc x Nf sub-sampling
            matrix defined by pi_{i;j} = delta_{sample}(i);j. The matrix has one non-zero (one)
            entry in each row

        Args:
            sample_indices (np.ndarray): samples drawn from the batch
            dimension (Tuple): dimensionality of the pi matrix N_c x N_f

        Returns:
            jax.numpy.ndarray: resulting pi matrix
        """
        # pi matrix is a N_c by N_f (or sample size times batch size) matrix with binary entries.
        # The element of the matrix is 1 when
        pi = np.zeros(dimension)

        for i in range(dimension[0]):
            pi[i, sample_indices[i]] = 1
            # pi[i][sample_indices[i]] = 1

        return pi
    
    @staticmethod
    def kernel_regression(kernel,  X_train: jax.numpy.ndarray, X_test: jax.numpy.ndarray, Y_train: jax.numpy.ndarray, regularization_lambda = 0.00001) -> jax.numpy.ndarray:
        # The data matrix (theta in the original paper)
        k_matrix = kernel(X_train, X_train, 'nngp')
        k_matrix += regularization_lambda * np.identity(k_matrix.shape[0])
        
        # The test matrix 
        t_matrix = kernel(X_test, X_train, 'nngp')
        
        # Regression coefficients in feature space
        # coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
        coeff, _, _, _ = np.linalg.lstsq(k_matrix, Y_train, rcond=1e-6)        
        prediction = np.matmul(t_matrix, coeff)
        
        return prediction