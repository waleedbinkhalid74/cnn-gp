import os
from pickletools import optimize
from typing import Tuple
import warnings

from scipy.linalg import lstsq
from cnn_gp import NNGPKernel
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

ACCEPTED_OPTIMIZERS = ['SGD', 'ADAM']

class KernelFlowsCNNGP():
    """Class to model Kernel Flows for convolutional neural network induced gaussian process kernels
        Pointwise operations (elementwise addition, multiplication, math functions - sin(), cos(), sigmoid() etc.)
        can be fused into a single kernel to amortize memory access time and kernel launch time.
        Pointwise operations are memory-bound, for each operation PyTorch launches a separate kernel.
        Each kernel loads data from the memory, performs computation (this step is usually inexpensive) and stores results back into the memory.


        For reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations
    """
    def __init__(self, cnn_gp_kernel: NNGPKernel, lr: float = 0.1,
                 beta: float = 0.9, regularization_lambda: float = 0.000001,
                 reduction_constant: float = 0.0):
        """Constructor for the Kernel Flow class that uses convolutional neural networks induced gaussian process kernels
           (Note, can also work with other kernels but they need to be a torch.nn.Module. See kernels folder for examples)

        Args:
            cnn_gp_kernel (NNGPKernel): cnn induced gp kernel
            lr (float, optional): learning rate. Defaults to 0.1.
            beta (float, optional): beta parameter. Defaults to 0.9.
            regularization_lambda (float, optional): regularization. Defaults to 0.000001.
            reduction_constant (float, optional): reduction constant. Defaults to 0.0.
        """
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
    def X(self) -> torch.Tensor:
        """Getter for training data

        Returns:
            torch.Tensor: Training Data
        """
        return self._X

    @X.setter
    def X(self, X: torch.Tensor):
        """Setter for training data

        Args:
            X (torch.Tensor): Training Data
        """
        self._X = X.to(torch.float32)

    @property
    def Y(self) -> torch.Tensor:
        """Getter for training labels

        Returns:
            torch.Tensor: training labels
        """
        return self._Y

    @Y.setter
    def Y(self, Y: torch.Tensor):
        """Setter for training labels

        Args:
            Y (torch.Tensor): training labels
        """
        self._Y = Y.to(torch.float32)

    @property
    def cnn_gp_kernel(self) -> NNGPKernel:
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
    def sample_selection(data_size: int, size: int) -> np.ndarray:
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
    def batch_creation(dataset_size:int, batch_size: int, sample_proportion: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
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


    @staticmethod
    def _block_kernel_eval(X: torch.Tensor, Y: torch.Tensor, blocksize: int, kernel: nn.Module) -> np.ndarray:
        """Evaluates the kernel matrix using a block wise evaluation approach. First the blocks are evaluated where the horizontal remainders (rh) are calculated in the same loop.
        The vertical remainders (rv) are evaluated in a separate loop.
        ---------------------
        |       |       |   |
        |   b   |  b    | rh|
        |       |       | rh|
        ---------------------
        |       |       |   |
        |   b   |   b   | rh|
        |       |       |   |
        ---------------------
        |   rv  |   rv  | rv|
        ---------------------
        Args:
            X (torch.Tensor): X input of kernel K(X, .)
            Y (torch.Tensor): Y input of kernel K(., Y)
            blocksize (int): Size of each block
            kernel (nn.Module): Callable function that evaluates the kernel

        Returns:
            np.ndarray: Evaluated kernel result
        """
        with torch.no_grad():
            kernel.eval()
            blocks_vertical = X.shape[0] // blocksize
            blocks_horizontal = Y.shape[0] // blocksize
            remainder_vertical = X.shape[0] - blocks_vertical*blocksize
            remainder_horizontal = Y.shape[0] - blocks_horizontal*blocksize
            block_horizontal  = 0
            k_matrix = np.ones((X.shape[0], Y.shape[0]), dtype=float)
            #  Handling main chunk of the blocks
            for block_vertical in tqdm(range(blocks_vertical)):
                X_batch_train_vertical = X[block_vertical*blocksize:(block_vertical+1)*blocksize]
                for block_horizontal in range(blocks_horizontal):
                    Y_batch_train_horizontal = Y[block_horizontal*blocksize:(block_horizontal+1)*blocksize]
                    k_matrix_batch = kernel(X_batch_train_vertical, Y_batch_train_horizontal)
                    k_matrix[block_vertical*blocksize:(block_vertical+1)*blocksize, block_horizontal*blocksize:(block_horizontal+1)*blocksize] = k_matrix_batch.detach().numpy()
                    del k_matrix_batch, Y_batch_train_horizontal
                # Handling horizontal remainders
                if remainder_horizontal > 0:
                    Y_batch_train_horizontal = Y[(block_horizontal+1)*blocksize:(block_horizontal+1)*blocksize + remainder_horizontal]
                    k_matrix_batch = kernel(X_batch_train_vertical, Y_batch_train_horizontal)
                    k_matrix[block_vertical*blocksize:(block_vertical+1)*blocksize, (block_horizontal+1)*blocksize:(block_horizontal+1)*blocksize + remainder_horizontal] = k_matrix_batch.detach().numpy()

            # Evaluating vertical remainder blocks rv
            if remainder_vertical > 0:
                X_batch_train_vertical = X[(block_vertical+1)*blocksize:(block_vertical+1)*blocksize + remainder_vertical]
                for block_horizontal in range(blocks_horizontal):
                    Y_batch_train_horizontal = Y[block_horizontal*blocksize:(block_horizontal+1)*blocksize]
                    k_matrix_batch = kernel(X_batch_train_vertical, Y_batch_train_horizontal)
                    k_matrix[(block_vertical+1)*blocksize:(block_vertical+1)*blocksize + remainder_vertical, block_horizontal*blocksize:(block_horizontal+1)*blocksize] = k_matrix_batch.detach().numpy()
                if remainder_horizontal > 0:
                    Y_batch_train_horizontal = Y[(block_horizontal+1)*blocksize:(block_horizontal+1)*blocksize + remainder_horizontal]
                    k_matrix_batch = kernel(X_batch_train_vertical, Y_batch_train_horizontal)
                    k_matrix[(block_vertical+1)*blocksize:(block_vertical+1)*blocksize + remainder_vertical, (block_horizontal+1)*blocksize:(block_horizontal+1)*blocksize + remainder_horizontal] = k_matrix_batch.detach().numpy()

        return k_matrix

    @staticmethod
    def kernel_regression(X_test: torch.Tensor, X_train: torch.Tensor,
                          Y_train: torch.Tensor, kernel: nn.Module, regularization_lambda = 0.0001,
                          blocksize: int = False, save_kernel: str = False) -> np.ndarray:
        """Applies Kernel regression to provided data

        Args:
            X_test (torch.Tensor): Test dataset
            X_train (torch.Tensor): Train dataset
            Y_train (torch.Tensor): Train dataset labels, for classification tasks must be in one hot encoding.
            kernel (nn.Module): Kernel used for kernel regression.
            regularization_lambda (float, optional): Regularization parameter. Defaults to 0.0001.
            blocksize (int, optional): Number of elements in each block. Defaults to False.

        Returns:
            np.ndarray: Prediction result
        """


        if blocksize is False:
            with torch.no_grad():
                k_matrix = kernel(X_train, X_train).detach().numpy()
                k_matrix += regularization_lambda * np.eye(k_matrix.shape[0])
                t_matrix = kernel(X_test, X_train).detach().numpy()
                prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train.detach().numpy()))
            return prediction
        elif blocksize > X_train.shape[0] or blocksize > Y_train.shape[0]:
            raise ValueError("Blocksize must be smaller or equal to the size of the input")



        # matrix block evaluation
        k_matrix = np.ones((X_train.shape[0], X_train.shape[0]), dtype=float)
        t_matrix = np.ones((X_test.shape[0], X_train.shape[0]), dtype=float)

        k_matrix = KernelFlowsCNNGP._block_kernel_eval(X=X_train,
                                                        Y=X_train,
                                                        blocksize=blocksize,
                                                        kernel=kernel)

        k_matrix += regularization_lambda * np.eye(k_matrix.shape[0])

        t_matrix = KernelFlowsCNNGP._block_kernel_eval(X=X_test,
                                                        Y=X_train,
                                                        blocksize=blocksize,
                                                        kernel=kernel)

        if save_kernel:
            np.save(os.getcwd() + '/saved_kernels/' + save_kernel + '_k_matrix.npy', k_matrix)
            np.save(os.getcwd() + '/saved_kernels/' + save_kernel + '_t_matrix.npy', t_matrix)

#########################VERIFY - REPLACING INVERSE WITH LEAST SQ AS PER MARTIN FOR NUMERICAL REASONS##################################
        # prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train))
        k_inv_Y, _, _, _ = lstsq(k_matrix, Y_train.detach().numpy(), cond=1e-8)
        prediction_lstsq = np.matmul(t_matrix, k_inv_Y)

#########################VERIFY - REPLACING INVERSE WITH LEAST SQ AS PER MARTIN FOR NUMERICAL REASONS##################################
        return prediction_lstsq#, k_matrix, t_matrix


        # T matrix block evaluation

    def sample_size_linear(self, iterations, range_tuple):
        return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]

    @staticmethod
    def pi_matrix(sample_indices: np.ndarray, dimension: Tuple) -> torch.Tensor:
        """Evaluates the pi matrix. pi matrix is the corresponding Nc x Nf sub-sampling
            matrix defined by pi_{i;j} = delta_{sample}(i);j. The matrix has one non-zero (one)
            entry in each row

        Args:
            sample_indices (np.ndarray): samples drawn from the batch
            dimension (Tuple): dimensionality of the pi matrix N_c x N_f

        Returns:
            torch.Tensor: resulting pi matrix
        """
        # pi matrix is a N_c by N_f (or sample size times batch size) matrix with binary entries.
        # The element of the matrix is 1 when
        pi = torch.zeros(dimension)

        for i in range(dimension[0]):
            pi[i][sample_indices[i]] = 1

        return pi

    def rho(self, X_batch: torch.Tensor, Y_batch: torch.Tensor,
            Y_sample: torch.Tensor, pi_matrix: torch.Tensor) -> torch.Tensor:
        """Calculates the rho which acts as the loss function for the Kernel Flow method. It evaluates how good the results were even when the
        training input was reduced by a factor.

        Args:
            X_batch (torch.Tensor): Training batch dataset
            Y_batch (torch.Tensor): Training batch dataset labels in one hot encoding
            Y_sample (torch.Tensor): Training sample dataset drawn from the batch in one hot encoding
            pi_matrix (torch.Tensor): pi matrix

        Returns:
            torch.Tensor: Resulting value of rho
        """
        # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)
        # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
        # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Calculate kernel theta = Kernel(X_Nf, X_Nf). NOTE: This is the most expensive step of the algorithm
        theta = self.cnn_gp_kernel(X_batch, X_batch)

        # Calculate sample_matrix = pi_mat*theta*pi_mat^T
        sample_matrix = torch.matmul(pi_matrix, torch.matmul(theta, torch.transpose(pi_matrix, 0, 1)))

        # Add regularization
        inverse_data = torch.linalg.inv(theta + self.regularization_lambda * torch.eye(theta.shape[0]))

        # Delete theta matrix to free memory as it is not needed beyond this point
        del theta

        inverse_sample = torch.linalg.inv(sample_matrix + self.regularization_lambda * torch.eye(sample_matrix.shape[0]))

        # Calculate numerator
        numerator = torch.matmul(torch.transpose(Y_sample,0,1), torch.matmul(inverse_sample, Y_sample))
        # Calculate denominator
        denominator = torch.matmul(torch.transpose(Y_batch,0,1), torch.matmul(inverse_data, Y_batch))
        # Calculate rho
        rho = 1 - torch.trace(numerator)/torch.trace(denominator)

        if torch.isnan(rho):
            raise ValueError("rho is NaN!")

        return rho #, sample_matrix, inverse_data, inverse_sample, numerator, denominator


    def fit(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size: int = False,
            sample_proportion: float = 0.5, optimizer: str = 'SGD', adaptive_size: bool = False):
        if optimizer not in ACCEPTED_OPTIMIZERS:
            raise RuntimeError("Optimizer should be a string in [SGD, ADAM]")
        # self.cnn_gp_kernel.train()
        # TODO: Incase N_I is the sample size that can be extracted for regression purposes,
        # we do not need to save the entire training set. Only the N_I sampled.
        # This will then have to be updated

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
            # X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]
            N_f = len(batch_indices)
            N_c = len(sample_indices)

            # Calculate pi matrix
            pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices, dimension=(N_c, N_f))

            # NOTE: The second code snippet does not zero the memory of each individual parameter, also the subsequent
            #       backward pass uses assignment instead of addition to store gradients, this reduces the number of memory operations.
            # For reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            optimizer.zero_grad()
            # for param in self.cnn_gp_kernel.parameters():
            #     param.grad = None

            # Calculate rho
            rho = self.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)
            if  rho > 1.01 or rho < -0.1:
                warnings.warn("Warning, rho outside [0,1]")
                print(f"""Warning, rho outside [0,1]. rho = {rho}""")

            if torch.isnan(rho):
                raise ValueError("rho is NaN!")
            # Calculate gradients
            rho.backward(create_graph=False)

            # Optimization step
            optimizer.step()

            # Store value of rho
            self.rho_values.append(rho.detach().numpy())

            del rho

    def predict(self, X_test: torch.Tensor, N_I: int = False) -> torch.Tensor:
        """Predict method for trained Kernel Flow model

        Args:
            X_test (torch.Tensor): Test dataset to predict
            N_I (int, optional): This specifies how much of the original training dataset should be used in the kernel regression.
            It is a good idea to set this parameter since often the training set is very large and evaluations can be prohibitively expensive. Defaults to False.

        Returns:
            torch.Tensor: Prediction results
        """
        self.cnn_gp_kernel.eval()
        if N_I is False:
            _, batch_indices = KernelFlowsCNNGP.batch_creation(dataset_size= self._X.shape[0],
                                                                    batch_size= self._X.shape[0])
        else:
            _, batch_indices = KernelFlowsCNNGP.batch_creation(dataset_size= self._X.shape[0],
                                                                    batch_size= N_I)

        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]

        del self._X
        del self._Y

        with torch.no_grad():
            prediction = KernelFlowsCNNGP.kernel_regression(X_train= X_batch,
                                                            Y_train= Y_batch,
                                                            X_test=X_test,
                                                            kernel=self.cnn_gp_kernel,
                                                            regularization_lambda=self.regularization_lambda)

        return prediction