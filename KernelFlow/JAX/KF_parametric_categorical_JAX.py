from functools import partial
from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad , random
from skopt import gp_minimize
from tqdm import tqdm
from scipy.optimize import OptimizeResult
from neural_tangents import stax
from KernelFlow.JAX.KF_base import KernelFlowsJAXBase

class KernelFlowsPJAX(KernelFlowsJAXBase):
    """This class is used to represent the Parametric Kernel Flows on NNGPs using the JAX library. This currently only supports classification tasks.

    Args:
        KernelFlowsJAXBase: Base class of parameteric Kernel Flows using JAX
    """
    def __init__(self, kernel_layers: int, kernel_activation: str = 'relu', kernel_output_dim: int = 1, lr: float = 0.1,
                 regularization_lambda: float = 0.00001,
                 reduction_constant: float = 0.0) -> None:
        """Initializes the kernel flows algorithm with user hyper parameter provided settings

        Args:
            kernel_layers (int): Number of layers in the Dense NNGP kernel
            kernel_activation (str, optional): Activation used in the NNGP kernel. Defaults to 'relu'.
            kernel_output_dim (int, optional): Output dimension of the NN that represents the GP kernel. Defaults to 1.
            lr (float, optional): Learning rate used in gradient based optimization. Defaults to 0.1.
            regularization_lambda (float, optional): Regularization coefficient. Defaults to 0.00001.
        """

        super().__init__()
        self.kernel_layers = kernel_layers
        if kernel_activation == 'relu':
            self.kernel_activation = stax.Relu()
        elif kernel_activation == 'erf':
            self.kernel_activation = stax.Erf()
        elif kernel_activation == 'sigmoid':
            self.kernel_activation = stax.Sigmoid_like()
        self.regularization_lambda = regularization_lambda
        self.kernel_output_dim = kernel_output_dim
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        self.learning_rate = lr
        self.regularization_lambda = regularization_lambda
        self.sigma_w = []
        self.sigma_b = []

    def make_kernel(self, W_std: float, b_std: float):
        """Constructs an NNGP kernel with the weight and bias standard deviation as provided by the user. The number of layers and the hidden layers
        are the same as initialized by the user in the constructor

        Args:
            W_std (float): Standard deviation of weights
            b_std (float): Standard deviation of biases

        Returns:
            Callable dense NNGP kernel
        """
        layers = []
        for i in range(self.kernel_layers - 1):
            layers += [stax.Dense(25, W_std=W_std, b_std=b_std), self.kernel_activation]

        init_fn, apply_fn, kernel_fn = stax.serial(*layers,
                    stax.Dense(self.kernel_output_dim, W_std=W_std, b_std=b_std))
        kernel_fn = jit(kernel_fn, static_argnums=(2,))
        return kernel_fn

    def _rho_bo(self, batch_size: int, sample_proportion: float, params: list) -> float:
        """Method to evaluate rho for the bayesian optimization step

        Args:
            batch_size (int): Batch size to select iid randomly for kernel flows
            sample_proportion (float): Sample proportion to select iid randomly from batch
            params (list): List containing tuples with the search range of each parameter used in bayesian optimization

        Returns:
            float: evaluted rho
        """
        sample_indices, batch_indices = KernelFlowsPJAX.batch_creation(dataset_size= self.X.shape[0],
                                                                batch_size= batch_size,
                                                                sample_proportion= sample_proportion)
        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]
        # X_sample = X_batch[sample_indices]
        Y_sample = Y_batch[sample_indices]
        N_f = len(batch_indices)
        N_c = len(sample_indices)


        # assert len(params) == len(list(self.cnn_gp_kernel.parameters()))
        # rho = 1 - trace(Y_s^T * K(X_s, X_s)^-1 * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Calculation of two kernels is expensive so we use proposition 3.2 Owhadi 2018
        # rho = 1 - trace(Y_s^T * (pi_mat * K(X_b, X_b)^-1 pi_mat^T) * Y_s) / trace(Y_b^T K(X_b, X_b)^-1 Y_b)

        # Set the parameters of the kernel
        kernel_fn = self.make_kernel(params[0], params[1])
        kernel_matrix = kernel_fn(X_batch, X_batch, 'nngp')
        kernel_matrix = np.array(kernel_matrix)
        # Calculate pi matrix
        pi = KernelFlowsPJAX.pi_matrix(sample_indices, (sample_indices.shape[0], X_batch.shape[0]))   
        sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
        
        Y_sample = Y_batch[sample_indices]
        
        inverse_data = np.linalg.inv(kernel_matrix + self.regularization_lambda * np.identity(kernel_matrix.shape[0]))
        inverse_sample = np.linalg.inv(sample_matrix + self.regularization_lambda * np.identity(sample_matrix.shape[0]))
    

        top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
        bottom = np.matmul(Y_batch.T, np.matmul(inverse_data, Y_batch))
        rho_val = 1 - np.trace(top)/np.trace(bottom)

        if rho_val <= 0.0:
            print("Warning, rho < 0. Setting to 1.")
            rho_val = 1.0
        # TODO: Check for accuracy
        eps = 1e-1
        min_rho = np.min(self.rho_values) if len(self.rho_values) > 0.0 else 0.0
        if rho_val < min_rho:
            rho_counter_check = self._rho_bo(batch_size=batch_size, sample_proportion=sample_proportion, params=params)
            if rho_counter_check > rho_val + eps:
                print("Value of rho was not stable, setting to 1.")
                rho_val = 1
        self.rho_values.append(rho_val)
        return rho_val

    def _rho_fd(self, X_batch: np.ndarray, Y_batch: np.ndarray,
            Y_sample: np.ndarray, pi_matrix: np.ndarray, params: list) -> float:
        """Evaluates the value of rho for the finite difference optimization method

        Args:
            X_batch (np.ndarray): Batch points selected from entire dataset by iid random sampling
            Y_batch (np.ndarray): Targets of batch points
            Y_sample (np.ndarray): Samples selected from the batch by iid random sampling
            pi_matrix (np.ndarray): pi matrix representing the delta matrix of which points from the batch were selected in the sample
            params (list): parameters of the NNGP kernel. First param is the standard deviation of the weight and second is the standard deviation of the bias.

        Returns:
            float: evaluated rho
        """

        # Set the parameters of the kernel
        kernel_fn = self.make_kernel(params[0], params[1])
        kernel_matrix = kernel_fn(X_batch, X_batch, 'nngp')
        kernel_matrix = np.array(kernel_matrix)
        sample_matrix = np.matmul(pi_matrix, np.matmul(kernel_matrix, np.transpose(pi_matrix)))
        inverse_data = np.linalg.inv(kernel_matrix + self.regularization_lambda * np.identity(kernel_matrix.shape[0]))
        inverse_sample = np.linalg.inv(sample_matrix + self.regularization_lambda * np.identity(sample_matrix.shape[0]))
        top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
        bottom = np.matmul(Y_batch.T, np.matmul(inverse_data, Y_batch))
        rho_val = 1 - np.trace(top)/np.trace(bottom)

        if rho_val <= 0.0:
            print("Warning, rho < 0.")
            # rho_val = 1.0
        return rho_val

    def fit_bayesian_optimization(self, X: np.ndarray, Y: np.ndarray, iterations: int , batch_size: int = False,
                                   sample_proportion: float = 0.5, parameter_bounds_BO: list= None, 
                                   random_starts: int = 15) -> OptimizeResult:
        """Applies the parameteric kernel flows algorithm on NNGPs using Bayesian Optimization as the optimization technique

        Args:
            X (np.ndarray): Training dataset
            Y (np.ndarray): Targets of training dataset
            iterations (int): Number of iterations to execute the kernel flows algorithm
            batch_size (int, optional): Size of the batch to select randomly iid from the training dataset. Defaults to False.
            sample_proportion (float, optional): Proportion of the batch to select randomly iid to form the sample. Defaults to 0.5.
            parameter_bounds_BO (list, optional): The range to search for optimized parameters given as a list of tuples. Defaults to None.
            random_starts (int, optional): Number of random starts before the acquisition function is used to calculate the next evaluation point. Defaults to 15.

        Returns:
            OptimizeResult: Object containing the optimization result and meta data
        """
        self.X = X
        self.Y = Y
        rho_objective = partial(self._rho_bo, batch_size, sample_proportion)
        if parameter_bounds_BO is None:
            no_params = 2 # Only global variance of weight and bias involved so total 2 params
            lower_bounds = list(0.0 * np.ones(no_params).astype(np.float64))
            upper_bounds = list(200.0*np.ones(no_params).astype(np.float64))
            parameter_bounds = list(zip(lower_bounds, upper_bounds))
        else:
            parameter_bounds = parameter_bounds_BO

        bo_result = gp_minimize(rho_objective,  # the function to minimize
                        parameter_bounds,   # the bounds on each parameter
                        acq_func="EI",      # the acquisition function
                        n_calls=iterations,         # the number of evaluations of f
                        n_random_starts=random_starts,  # the number of random initialization points
                        # noise=0.1**2,       # the noise level (optional)
                        random_state=1234)  # the random seed

        self.optimized_kernel = self.make_kernel(bo_result.x[0], bo_result.x[1])
        return bo_result
    
    def fit_finite_difference(self, X: np.ndarray, Y: np.ndarray, iterations: int , init_sigma_w: float, init_sigma_b: float,
                                batch_size: int = False, sample_proportion: float = 0.5, 
                                h: float = 1e-4) -> list:
        """Applies the parameteric kernel flows algorithm on NNGPs using finite difference to approximate the gradient and then applies gradient based optimization.

        Args:
            X (np.ndarray): Training dataset
            Y (np.ndarray): Targets of training dataset
            iterations (int): Number of iterations to execute the kernel flows algorithm
            batch_size (int, optional): Size of the batch to select randomly iid from the training dataset. Defaults to False.
            sample_proportion (float, optional): Proportion of the batch to select randomly iid to form the sample. Defaults to 0.5.
            init_sigma_w (float): initial value of the variance of weights
            init_sigma_b (float): initial value of the variance of biases
            h (float, optional): Perturbation used in the forward difference approximation of gradients. Defaults to 1e-4.

        Returns:
            list: rho values
        """
        self.X = X
        self.Y = Y
        sigma_w = init_sigma_w
        sigma_b = init_sigma_b
        for i in tqdm(range(iterations)):
            self.sigma_w.append(sigma_w)
            self.sigma_b.append(sigma_b)
            
            # Create batch N_f and sample N_c = p*N_f
            sample_indices, batch_indices = KernelFlowsPJAX.batch_creation(dataset_size= self.X.shape[0],
                                                                            batch_size= batch_size,
                                                                            sample_proportion= sample_proportion)
            X_batch = self.X[batch_indices]
            Y_batch = self.Y[batch_indices]
            # X_sample = X_batch[sample_indices]
            Y_sample = Y_batch[sample_indices]
            N_f = len(batch_indices)
            N_c = len(sample_indices)

            # Calculate pi matrix
            pi = KernelFlowsPJAX.pi_matrix(sample_indices, (sample_indices.shape[0], X_batch.shape[0]))   

            # NOTE: Number of forward passes = No_parameters + 1 per iteration!
            # For every parameter with torch.no_grad()
                # get rho(w_i + dw/2) rho = self.rho(...) This can be changed to forward or backward differences as well
                # get rho(w_i - dw/2) rho = self.rho(...) This can be changed to forward or backward differences as well
                # d(rho)/dw_i = (rho(w_i + dw/2) - rho(w_i - dw/2)) / dw
            # Apply update step in the end
            rho = self._rho_fd(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, 
                        pi_matrix=pi, params=[sigma_w, sigma_b])
            self.rho_values.append(rho)
            # Applying finite difference
            rho_dw = self._rho_fd(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, 
                        pi_matrix=pi, params=[sigma_w + h, sigma_b])
            rho_db = self._rho_fd(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, 
                        pi_matrix=pi, params=[sigma_w, sigma_b + h])
            drho_dw = (rho_dw - rho) / h
            drho_db = (rho_db - rho) / h
            sigma_b -= self.learning_rate*drho_db
            sigma_w -= self.learning_rate*drho_dw

        self.optimized_kernel = self.make_kernel(sigma_w, sigma_b)
        return self.rho_values