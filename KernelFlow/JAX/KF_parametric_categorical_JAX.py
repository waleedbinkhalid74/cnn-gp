from functools import partial
from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad , random
from skopt import gp_minimize
from tqdm import tqdm
import jax
from neural_tangents import stax
from KernelFlow.JAX.KF_base import KernelFlowsJAXBase

class KernelFlowsPJAX(KernelFlowsJAXBase):
    def __init__(self, kernel_layers, kernel_activation = 'relu', kernel_output_dim = 1, lr: float = 0.1,
                 beta: float = 0.9, regularization_lambda: float = 0.00001,
                 reduction_constant: float = 0.0) -> None:

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
        self.beta = beta
        self.reduction_constant = reduction_constant
        self.regularization_lambda = regularization_lambda

    def make_kernel(self, W_std, b_std):
        layers = []
        for i in range(self.kernel_layers - 1):
            layers += [stax.Dense(25, W_std=W_std, b_std=b_std), self.kernel_activation]

        init_fn, apply_fn, kernel_fn = stax.serial(*layers,
                    stax.Dense(self.kernel_output_dim, W_std=W_std, b_std=b_std))
        kernel_fn = jit(kernel_fn, static_argnums=(2,))
        return kernel_fn

    def _rho_bo(self, batch_size, sample_proportion, params, store_rho: bool = True):
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
        return rho_val

    def fit_bayesian_optimization(self, X, Y, iterations: int , batch_size: int = False,
                                   sample_proportion: float = 0.5, parameter_bounds_BO: list= None, 
                                   random_starts: int = 15):
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
    