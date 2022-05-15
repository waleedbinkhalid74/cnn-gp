import itertools
import os
from typing import Tuple
import warnings
from xmlrpc.client import boolean
from torch.multiprocessing import Process, Pool, set_start_method, Queue
from scipy.linalg import lstsq
from zmq import device
from cnn_gp import NNGPKernel, ProductIterator
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from functools import partial
from skopt import gp_minimize
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

class tqdm_skopt(object):

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res=1):
        self._bar.update(1)
        # if len(res.x_iters) % 5 == 0:
        #     print(f"""On iteration {len(res.x_iters)}""")

ACCEPTED_OPTIMIZERS = ['SGD', 'ADAM']

def kernel_wrapper(X, Y, queue, kernel):
    # async with torch.no_grad:
    torch.set_grad_enabled(False)
    result = kernel(X,Y)
    queue.put(result)

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
                 reduction_constant: float = 0.0, blocksize: int = 200,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
        self.block_size = blocksize # Number of images to process at once when evaluating the Kernel
        self.device = device

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
    def _block_kernel_eval_parallel(X: torch.Tensor, Y: torch.Tensor, block_size: int, kernel: nn.Module) -> np.ndarray:

        # NOTE: For reference: https://pytorch.org/docs/stable/notes/multiprocessing.html
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        queue = Queue()
        X_blocks = torch.split(X, block_size)
        Y_blocks = torch.split(Y, block_size)
        block_pairs = list(itertools.product(X_blocks, Y_blocks))
        processes = []
        results = []
        for block in block_pairs:
            arguments = block + (queue, kernel)
            process = Process(target=kernel_wrapper, args= arguments)
            processes.append(process)
            process.start()
        for i in range(len(processes)):
            results.append(queue.get())
        for proc in processes:
            proc.join()

        results = torch.cat(results).numpy()
        results_reformated = np.zeros((X.shape[0], Y.shape[0]))
        counter = 0
        for i in range(len(X_blocks)):
            for j in range(len(Y_blocks)):
                results_reformated[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = results[counter*block_size:(counter+1)*block_size]
                counter += 1
        return results_reformated


    @staticmethod
    def _block_kernel_eval(X: torch.Tensor, Y: torch.Tensor, blocksize: int,
                           kernel: nn.Module, worker_rank: int=0, n_workers: int=1) -> torch.Tensor:
        """Evaluates the kernel matrix using a block wise evaluation approach.
        Args:
            X (torch.Tensor): X input of kernel K(X, .)
            Y (torch.Tensor): Y input of kernel K(., Y)
            blocksize (int): Size of each block
            kernel (nn.Module): Callable function that evaluates the kernel

        Returns:
            torch.Tensor: Evaluated kernel result
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k_matrix = torch.ones((X.shape[0], Y.shape[0]), dtype=torch.float32).to(device)
        it = ProductIterator(blocksize, X, Y, worker_rank=worker_rank, n_workers=n_workers)
        for same, (i, x), (j, y) in it:
            k = kernel(x, y, same, diag=False)
            k_matrix[i:i+len(x), j:j+len(y)] = k

        return k_matrix


    @staticmethod
    def kernel_regression(X_test: torch.Tensor, X_train: torch.Tensor,
                          Y_train: torch.Tensor, kernel: nn.Module, regularization_lambda = 0.0001,
                          blocksize: int = 200, save_kernel: str = False,
                          worker_rank:int = 0, n_workers: int=1) -> np.ndarray:
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
        # matrix block evaluation
        kwargs = dict(worker_rank=worker_rank, n_workers=n_workers)

        with torch.no_grad():
            k_matrix = KernelFlowsCNNGP._block_kernel_eval(X=X_train,
                                                            Y=X_train,
                                                            blocksize=blocksize,
                                                            kernel=kernel,
                                                            **kwargs)

            t_matrix = KernelFlowsCNNGP._block_kernel_eval(X=X_test,
                                                            Y=X_train,
                                                            blocksize=blocksize,
                                                            kernel=kernel,
                                                            **kwargs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k_matrix += regularization_lambda * torch.eye(k_matrix.shape[0]).to(device=device)


        if save_kernel:
            torch.save(os.getcwd() + '/saved_kernels/' + save_kernel + '_k_matrix.pt', k_matrix)
            torch.save(os.getcwd() + '/saved_kernels/' + save_kernel + '_t_matrix.pt', t_matrix)

#########################VERIFY - REPLACING INVERSE WITH LEAST SQ AS PER MARTIN FOR NUMERICAL REASONS##################################
        prediction = torch.matmul(t_matrix, torch.matmul(torch.linalg.inv(k_matrix), Y_train))
        # k_inv_Y, _, _, _ = lstsq(k_matrix, Y_train.detach().numpy(), cond=1e-8)
        # prediction_lstsq = np.matmul(t_matrix, k_inv_Y)
#########################VERIFY - REPLACING INVERSE WITH LEAST SQ AS PER MARTIN FOR NUMERICAL REASONS##################################
        return prediction, k_matrix, t_matrix


    def sample_size_linear(self, iterations, range_tuple):
        return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]

    @staticmethod
    def _pi_matrix(pi_matrix:torch.Tensor, sample_indices: np.ndarray, dimension: Tuple) -> torch.Tensor:
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
        for i in range(dimension[0]):
            pi_matrix[i][sample_indices[i]] = 1

        return pi_matrix

    def pi_matrix(self, sample_indices: np.ndarray, dimension: Tuple):
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
        pi = torch.zeros(dimension).to(self.device)
        pi = KernelFlowsCNNGP._pi_matrix(pi_matrix=pi, dimension=dimension, sample_indices=sample_indices)
        return pi
    def _rho_bo(self, batch_size, sample_proportion, params) -> torch.Tensor:

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

        assert len(params) == len(list(self.cnn_gp_kernel.parameters()))
        # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
        # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Set the parameters of the kernel
        for i, param in enumerate(self.cnn_gp_kernel.parameters()):
            param.data = torch.tensor([params[i]]).to(self.device)

        # Calculate kernel theta = Kernel(X_Nf, X_Nf). NOTE: This is the most expensive step of the algorithm
        with torch.no_grad():
            theta = KernelFlowsCNNGP._block_kernel_eval(X=X_batch,Y=X_batch,kernel=self.cnn_gp_kernel,
                                                blocksize=self.block_size, worker_rank=0, n_workers=1)
        # theta = self.cnn_gp_kernel(X_batch, X_batch)

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

        # skopt Bayesian optimization requires a scalar output from the blackbox function
        self.rho_values.append(rho.numpy().item())
        # print(f"""The value of rho is {rho.numpy().item()}""")
        return rho.numpy().item()

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
        theta = KernelFlowsCNNGP._block_kernel_eval(X=X_batch,Y=X_batch,kernel=self.cnn_gp_kernel,
                                            blocksize=self.block_size, worker_rank=0, n_workers=1)
        # theta = self.cnn_gp_kernel(X_batch, X_batch)

        # Calculate sample_matrix = pi_mat*theta*pi_mat^T
        sample_matrix = torch.matmul(pi_matrix, torch.matmul(theta, torch.transpose(pi_matrix, 0, 1)))

        # Add regularization
        inverse_data = torch.linalg.inv(theta + self.regularization_lambda * torch.eye(theta.shape[0]).to(self.device))

        # Delete theta matrix to free memory as it is not needed beyond this point
        del theta

        inverse_sample = torch.linalg.inv(sample_matrix + self.regularization_lambda * torch.eye(sample_matrix.shape[0]).to(self.device))

        # Calculate numerator
        numerator = torch.matmul(torch.transpose(Y_sample,0,1), torch.matmul(inverse_sample, Y_sample))
        # Calculate denominator
        denominator = torch.matmul(torch.transpose(Y_batch,0,1), torch.matmul(inverse_data, Y_batch))
        # Calculate rho
        rho = 1 - torch.trace(numerator)/torch.trace(denominator)

        return rho

    def _fit_autograd(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size: int = False,
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
                                                                            sample_proportion= sample_size)
            X_batch = self.X[batch_indices]
            Y_batch = self.Y[batch_indices]
            # X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]
            N_f = len(batch_indices)
            N_c = len(sample_indices)

            # Calculate pi matrix
            pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices, dimension=(N_c, N_f))

            optimizer.zero_grad()

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

    def _fit_finite_difference(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size: int = False,
            sample_proportion: float = 0.5, optimizer: str = 'SGD', adaptive_size: bool = False, dw: float = 1e-4):

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
                                                                            sample_proportion= sample_size)
            X_batch = self.X[batch_indices]
            Y_batch = self.Y[batch_indices]
            # X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]
            N_f = len(batch_indices)
            N_c = len(sample_indices)

            # Calculate pi matrix
            pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=sample_indices, dimension=(N_c, N_f))
            optimizer.zero_grad()

            # NOTE: Number of forward passes = No_parameters + 1 per iteration!
            # For every parameter with torch.no_grad()
                # get rho(w_i + dw/2) rho = self.rho(...) This can be changed to forward or backward differences as well
                # get rho(w_i - dw/2) rho = self.rho(...) This can be changed to forward or backward differences as well
                # d(rho)/dw_i = (rho(w_i + dw/2) - rho(w_i - dw/2)) / dw
            # Apply update step in the end
            with torch.no_grad():
                rho = self.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample,
                               pi_matrix=pi_matrix)
            # rho_control = self.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)
            # rho_control.backward()

            if  rho > 1.01 or rho < -0.1:
                warnings.warn("Warning, rho outside [0,1]")
                print(f"""Warning, rho outside [0,1]. rho = {rho}""")

            if torch.isnan(rho):
                raise ValueError("rho is NaN!")

            # Calculate gradients using finite difference
            for param in self.cnn_gp_kernel.parameters():
                # rho = self.rho()
                # We do not wish to track gradients as we approximate them using finite differences
                with torch.no_grad():
                    old_param = param.data
                    param.data = param.data + dw
                    rho_dw = self.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample,
                                      pi_matrix=pi_matrix)
                    drho_dw = (rho_dw - rho) / dw
                    param.data = old_param
                    param.grad = drho_dw.unsqueeze(0).to(param.device)
            # Optimization step
            optimizer.step()

            # Store value of rho
            self.rho_values.append(rho.detach().numpy())

            del rho

    def _fit_bayesian_optimization(self, X: torch.Tensor, Y: torch.Tensor,
                                   iterations: int , batch_size: int = False,
                                   sample_proportion: float = 0.5,
                                   parameter_bounds_BO: list= None, random_starts: int = 15):

        self.X = X
        self.Y = Y
        # Since this is Bayesian optimization we do not need any iterations

        # Strip the arguments other than parameters since gp_minimize does not accept auxilary arguments as of yet.
        # This is an open issue Reference: https://github.com/scikit-optimize/scikit-optimize/issues/240
        # TODO: Once the args feature is added this can be made cleaner.
        rho_objective = partial(self._rho_bo, batch_size, sample_proportion)

        # QUESTION: Can the bounds be negative?
        if parameter_bounds_BO is None:
            no_params = len(list(self.cnn_gp_kernel.parameters()))
            lower_bounds = list(-10 * np.ones(no_params))
            upper_bounds = list(10.0*np.ones(no_params))
            parameter_bounds = list(zip(lower_bounds, upper_bounds))
        else:
            parameter_bounds = parameter_bounds_BO
        # Applied gaussian process based bayesian optimization
        bo_result = gp_minimize(rho_objective,  # the function to minimize
                        parameter_bounds,   # the bounds on each parameter
                        acq_func="EI",      # the acquisition function
                        n_calls=iterations,         # the number of evaluations of f
                        n_random_starts=random_starts,  # the number of random initialization points
                        # noise=0.1**2,       # the noise level (optional)
                        random_state=1234,  # the random seed
                        callback=[tqdm_skopt(total=iterations, desc="Bayesian Optimization")])

        new_parameters = bo_result.x
        for i, param in enumerate(self.cnn_gp_kernel.parameters()):
            param.data = torch.tensor([new_parameters[i]])

        return self.cnn_gp_kernel

    def fit(self, X: torch.Tensor, Y: torch.Tensor, iterations: int, batch_size: int = False,
            sample_proportion: float = 0.5, method:str = 'autograd', optimizer: str = 'SGD',
            dw: float = 0.001, adaptive_size: bool = False, parameter_bounds_BO: list = None,
            random_starts_BO: int = 15):
        """Fits the Kernel with optimial hyperparameters based on the Kernel Flow algorithm

        Args:
            X (torch.Tensor): Training dataset values
            Y (torch.Tensor): Training dataset labels
            iterations (int): Number of iterations
            batch_size (int, optional): Batch size N_f. Defaults to False.
            sample_proportion (float, optional): Proportion of the batch against which rho is calculated. Defaults to 0.5.
            method (str, optional): Differenciation method. Defaults to 'autograd'.
            optimizer (str, optional): Optimization technique. Defaults to 'SGD'.
            dw (float, optional): Increment perturbation for finite difference. Not needed for autograd. Defaults to 0.001.
            adaptive_size (bool, optional): If sample proportion should change with iterations. Defaults to False.

        Raises:
            ValueError: Incase incorrect differenciation strategy is selected
        """
        if method == 'autograd':
            self._fit_autograd(X=X, Y=Y, iterations=iterations, batch_size=batch_size,
                               sample_proportion=sample_proportion, optimizer=optimizer,
                               adaptive_size=adaptive_size)
        elif method == 'finite difference':
            self._fit_finite_difference(X=X, Y=Y, iterations=iterations, batch_size=batch_size,
                                        sample_proportion=sample_proportion,
                                        optimizer=optimizer, dw=dw, adaptive_size=adaptive_size)
        elif method == 'bayesian optimization':
            self._fit_bayesian_optimization(X=X, Y=Y, batch_size=batch_size,
                                            iterations=iterations,
                                            sample_proportion=sample_proportion,
                                            parameter_bounds_BO=parameter_bounds_BO,
                                            random_starts=random_starts_BO)
        else:
            raise ValueError("Method not understood. Please use either 'autograd', 'finite difference' or 'bayesian optimization'. Defaults to 'autograd'")

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

        prediction = KernelFlowsCNNGP.kernel_regression(X_train= X_batch,
                                                        Y_train= Y_batch,
                                                        X_test=X_test,
                                                        kernel=self.cnn_gp_kernel,
                                                        regularization_lambda=self.regularization_lambda)

        return prediction