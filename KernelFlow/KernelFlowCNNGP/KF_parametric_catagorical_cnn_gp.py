from typing import Tuple
from cnn_gp import NNGPKernel
import torch
import numpy as np
from tqdm import tqdm

class KernelFlowsCNNGP():
    """Class to model Kernel Flows for convolutional neural network induced gaussian process kernels
    """
    def __init__(self, cnn_gp_kernel: NNGPKernel, lr: float = 0.1,
                 beta: float = 0.9, regularization_lambda: float = 0.0001,
                 reduction_constant: float = 0.0):
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        self._cnn_gp_kernel = cnn_gp_kernel
        self.LR = lr
        self.beta = beta
        self.reduction_constant = reduction_constant
        self.regu_lambda = regularization_lambda

        self.X: torch.Tensor = None
        self.Y: torch.Tensor = None

    @property
    def cnn_gp_kernel(self):
        """Return the cnn induced gp for the kernel flow run

        Returns:
            NNGPKernel: cnn induced gp kernel
        """
        return self._cnn_gp_kernel

    @cnn_gp_kernel.setter
    def cnn_gp_kernel(self, cnn_gp_kernel: NNGPKernel):
        """Set the kernel to be trained using kernel flow

        Args:
            cnn_gp_kernel (NNGPKernel): Convolutional Neural Network induced Gaussian Process

        Raises:
            TypeError: Ensure model is of type CNN induced GP
        """
        if isinstance(cnn_gp_kernel, NNGPKernel):
            self._cnn_gp_kernel = cnn_gp_kernel
        else:
            raise TypeError("Kernel type should be NNGPKernel")

    @staticmethod
    def sample_selection(data_size: int, size: int):
        """Selects an iid sample from the dataset without replacement

        Args:
            data_size (int): size of data to sample from
            size (int): number of items to sample

        Returns:
            np.array: sampled indices
        """

        indices = np.arange(data_size)
        sample_indices = np.sort(np.random.choice(indices, size, replace= False))
        return sample_indices


    @staticmethod
    def batch_creation(dataset_size:int, batch_size: int, sample_proportion: float):
        """Creates a batch N_f and sample N_c from available data for kernel regression

        Args:
            dataset_size (int): size of entire dataset from which to create batches
            batch_size (int): N_f samples from entire batch
            sample_proportion (float): N_c samples from entire batch where N_c < N_f

        Returns:
            (np.array, np.array):  sample and batch indices respectively
        """
        # Error handling
        if batch_size > dataset_size:
            raise Exception("Batch size must be lesser than dataset size.")

        if sample_proportion > 1.0 or sample_proportion < 0.0:
            raise Exception("Sample proportion should be between 0 and 1")

        # batch creation
        if batch_size == False:
            batch_indices = np.arange(dataset_size)
        elif 0 < batch_size <= 1:
            batch_size = int(dataset_size * batch_size)
            batch_indices = KernelFlowsCNNGP.sample_selection(dataset_size, batch_size)
            data_batch = len(batch_indices)
        else:
            batch_size = int(batch_size)
            batch_indices = KernelFlowsCNNGP.sample_selection(dataset_size, batch_size)
            data_batch = len(batch_indices)

        # Sample from the mini-batch
        sample_size = int(np.ceil(data_batch*sample_proportion))
        sample_indices = KernelFlowsCNNGP.sample_selection(data_batch, sample_size)

        return sample_indices, batch_indices

    def kernel_regression(self):
        pass

    def sample_size_linear(self, iterations, range_tuple):
        return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]

    @staticmethod
    def pi_matrix(sample_indices, dimension: Tuple):
        # pi matrix is a N_c by N_f (or sample size times batch size) matrix with binary entries.
        # The element of the matrix is 1 when
        pi = torch.zeros(dimension)

        for i in range(dimension[0]):
            pi[i][sample_indices[i]] = 1

        return pi

    def rho(self, X_batch: torch.Tensor, Y_batch: torch.Tensor,
            X_sample: torch.Tensor, Y_sample: torch.Tensor):

        # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)
        # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
        # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)
        # Note: Using the notation of Owhadi 2018

        # Calculate kernel theta = Kernel(X_Nf, X_Nf)
        theta = self.cnn_gp_kernel(X_batch, X_batch)

        # TODO Calculate pi matrix

        # Calculate (pi_mat*theta*pi_mat^T)^-1

        # inverse_data = np.linalg.inv(theta + lambda_term * np.identity(theta.shape[0]))
        # inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))

        # Calculate numerator
        # top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
        # Calculate denominator
        # bottom = np.matmul(Y_batch.T, np.matmul(inverse_batch, Y_batch))
        # Calculate rho
        # rho = 1 - np.trace(top)/np.trace(bottom)

        pass

    def fit(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size = False,
            sample_proportion: float = 0.5, optimizer: str = 'SGD', adaptive_size: bool = False):
        # TODO: Incase N_I is the sample size that can be extracted for regression purposes,
        # we do not need to save the entire training set. Only the N_I sampled.
        # This will then have to be updated
        self.X = X
        self.Y = Y

        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0

        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = self.proportion
        elif adaptive_size == "Linear":
            sample_size_array = self.sample_size_linear(iterations, self.adaptive_range)
        else:
            print("Sample size not recognized")

        for i in tqdm(range(iterations)):
            if adaptive_size == "Linear":
                sample_size = sample_size_array[i]
            elif adaptive_size == "Dynamic" and adaptive_counter == 100:
                if adaptive_mean != 0:
                    change = np.mean(rho_100) - adaptive_mean
                else:
                    change = 0
                adaptive_mean = np.mean(rho_100)
                rho_100 = []
                sample_size += change - self.reduction_constant
                adaptive_counter= 0


            # Create batch N_f and sample N_c = p*N_f
            sample_indices, batch_indices = KernelFlowsCNNGP.batch_creation(dataset_size= self.X.shape[0], batch_size= batch_size, sample_proportion= sample_proportion)
            X_batch = self.X[batch_indices]
            Y_batch = self.Y[batch_indices]
            X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]

            # TODO: Calculate rho

            # TODO: Calculate gradients
            # TODO: Optimization step
    def predict(self):
        pass