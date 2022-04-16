from pickletools import optimize
from typing import Tuple
import warnings
from cnn_gp import NNGPKernel
import torch
import numpy as np
from tqdm import tqdm

ACCEPTED_OPTIMIZERS = ['SGD', 'ADAM']

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
        self.learning_rate = lr
        self.beta = beta
        self.reduction_constant = reduction_constant
        self.regularization_lambda = regularization_lambda

        self._X: torch.Tensor = None
        self._Y: torch.Tensor = None

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X.to(torch.float32)

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):
        self._Y = Y.to(torch.float32)

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

    def kernel_regression(self, X_test: torch.Tensor):
        if self._X is None or self.Y is None:
            raise Exception("Train dataset not provided.")

        k_matrix = self.cnn_gp_kernel(self.X, self.X)
        k_matrix += self.regularization_lambda * torch.eye(k_matrix.shape[0])
        t_matrix = self.cnn_gp_kernel(X_test, self.X)
        prediction = torch.matmul(t_matrix, torch.matmul(torch.linalg.inv(k_matrix), self.Y))
        return prediction

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
            Y_sample: torch.Tensor, pi_matrix: torch.Tensor):
        # TODO: Add docstrings
        # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)
        # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
        # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Calculate kernel theta = Kernel(X_Nf, X_Nf)
        theta = self.cnn_gp_kernel(X_batch, X_batch)

        # Calculate sample_matrix = pi_mat*theta*pi_mat^T
        sample_matrix = torch.matmul(pi_matrix, torch.matmul(theta, torch.transpose(pi_matrix, 0, 1)))

        # Add regularization
        inverse_data = torch.linalg.inv(theta + self.regularization_lambda * torch.eye(theta.shape[0]))
        inverse_sample = torch.linalg.inv(sample_matrix + self.regularization_lambda * torch.eye(sample_matrix.shape[0]))

        # Calculate numerator
        numerator = torch.matmul(torch.transpose(Y_sample,0,1), torch.matmul(inverse_sample, Y_sample))
        # Calculate denominator
        denominator = torch.matmul(torch.transpose(Y_batch,0,1), torch.matmul(inverse_data, Y_batch))
        # Calculate rho
        rho = 1 - torch.trace(numerator)/torch.trace(denominator)

        return rho


    def fit(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size = False,
            sample_proportion: float = 0.5, optimizer: str = 'SGD', adaptive_size: bool = False):
        # TODO: Incase N_I is the sample size that can be extracted for regression purposes,
        # we do not need to save the entire training set. Only the N_I sampled.
        # This will then have to be updated

        if optimizer not in ACCEPTED_OPTIMIZERS:
            raise RuntimeError("Optimizer should be a string in [SGD, ADAM]")

        # self.cnn_gp_kernel.train()

        if optimizer == 'SGD':
            optimizer =  torch.optim.SGD(self.cnn_gp_kernel.parameters(), lr=self.learning_rate)
        elif optimizer == 'ADAM':
            optimizer =  torch.optim.Adam(self.cnn_gp_kernel.parameters(), lr=self.learning_rate)

        self.X = X
        self.Y = Y

        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0

        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = sample_proportion
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
            sample_indices, batch_indices = KernelFlowsCNNGP.batch_creation(dataset_size= self._X.shape[0],
                                                                            batch_size= batch_size,
                                                                            sample_proportion= sample_proportion)
            X_batch = self.X[batch_indices]
            Y_batch = self.Y[batch_indices]
            X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]
            N_f = len(batch_indices)
            N_c = len(sample_indices)

            # Calculate pi matrix
            pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices, dimension=(N_c, N_f))

            # Calculate rho
            rho = self.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)
            if  rho > 1.01 or rho < -0.1:
                warnings.warn("Warning, rho outside [0,1]: ")
            # Calculate gradients
            optimizer.zero_grad()
            rho.backward()

            # Optimization step
            optimizer.step()

            # Store value of rho
            self.rho_values.append(rho.detach().numpy())

    def predict(self, X_test: torch.Tensor):
        self.cnn_gp_kernel.eval()
        with torch.no_grad():
            prediction = self.kernel_regression(X_test=X_test)

        return prediction