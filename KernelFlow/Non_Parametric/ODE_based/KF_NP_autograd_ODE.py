import autograd.numpy as np
from tqdm import tqdm
from .kernel_functions import kernels_dic
from .nabla_functions import nabla_dic
from scipy import integrate
import torch
from .auxilary_functions import *
#%%

class KernelFlowsNP_Autograd_ODE():
    
    def __init__(self, kernel_keyword, parameters, regression_type = "single"):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.points_hist = []
        self.batch_hist = []
        self.perturbation = []
        self.X_norm = []
        # Variables needed for flow transform --> This is a rough implementation geared towards experimentation
        self.X_batch = []
        self.Y_batch = []
        self.sample_indices = []

    def G(self, t: list, X: np.ndarray, Y: np.ndarray, batch_indices: np.ndarray, sample_indices: np.ndarray, not_batch: np.ndarray) -> np.ndarray:
        """A callable function that calculates the perturbation using the Frechet derivative of rho. The function can be used as a callable in the ODE solver provided by python.

        Args:
            t (list): Tiem interval to integrate the ODE on
            X (np.ndarray): X dataset
            Y (np.ndarray): Y target of dataset
            batch_indices (np.ndarray): Batch indices
            sample_indices (np.ndarray): Sample indices selected from batch randomly without replacement
            not_batch (np.ndarray): Indices not part of the batch

        Returns:
            np.ndarray: Preturbations calculated by frechet derivative for items in batch and by interpolation for items outside of batch
        """

        X = np.reshape(X, (Y.shape[0], X.shape[0] // Y.shape[0]))
        X_norm = np.linalg.norm(X)
        # print(X_norm)
        self.X_norm.append(X_norm)
        X_batch = X[batch_indices]
        Y_batch = Y[batch_indices]

        rho, g = grad_rho(self.parameters, X_batch, Y_batch, 
                    sample_indices, self.kernel_keyword, reg = self.regularization_lambda)
        g = -g
        if rho >1 or rho <0:
            print ("Rho outside allowed bounds", rho.item())
        g_interpolate, coeff = kernel_regression(X_batch, X[not_batch], g, self.parameters, self.kernel_keyword, regu_lambda = self.regularization_lambda)
        perturbation = np.zeros(X.shape)
        perturbation[batch_indices] = g
        perturbation[not_batch] = g_interpolate
        # perturbation = np.tanh(perturbation)
        self.perturbation.append(np.copy(perturbation))

        self.rho_values.append(rho.item())
        return perturbation.ravel()


    def fit(self, X: np.ndarray, Y: np.ndarray, iterations: int, batch_size: int, learning_rate:float = 0.1, type_epsilon: str = "relative", record_hist: bool = True, reg: float = 0.000001) -> np.ndarray:
        """Fit method to optimize a given kernel using non-parametric kernel flows using an ODE solver step for the updating of the datapoints

        Args:
            X (np.ndarray): Training dataset
            Y (np.ndarray): Training dataset targets
            iterations (int): Number of iterations to execute 
            batch_size (int): Batch size in each iteration to be choosen randomly without replacement
            learning_rate (float, optional): Not used in this method To be deleted. Defaults to 0.1.
            type_epsilon (str, optional): Not used in this method to be deleted. Defaults to "relative".
            record_hist (bool, optional): Not used in this method to be deleted. Defaults to True.
            reg (float, optional): Regularization. Defaults to 0.000001.

        Returns:
            np.ndarray: Perturbed training dataset
        """
        
        # Create a copy of the parameters (so the original parameters aren't modified)

        self.batch_size = batch_size
        if isinstance(X, torch.Tensor):
            X = X.numpy()
            Y = Y.numpy()
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        self.iteration = iterations
        self.points_hist.append(np.copy(X))
        self.regularization_lambda = reg
        
        if batch_size == False:
            self.regression = False
        
        self.X = np.copy(X)
        data_set_ind = np.arange(X.shape[0])
        for i in tqdm(range(iterations)):
            sample_indices, batch_indices = batch_creation(X, batch_size)
            self.X_batch.append(np.copy(X[batch_indices]))
            self.Y_batch.append(np.copy(Y[batch_indices]))
            self.batch_hist.append(np.copy(batch_indices))

            self.sample_indices.append(np.copy(sample_indices))
            # The indices of all the elements not in the batch
            not_batch = np.setdiff1d(data_set_ind, batch_indices)
            solution = integrate.solve_ivp(self.G, [0, 1.0],  X.ravel(), args=(Y, batch_indices, sample_indices, not_batch), method='RK45', rtol=1e-8, atol=1e-8)#, max_step=0.01) # Also tried Radau
            X = solution.y[:,-1]
            X = np.reshape(X, (Y.shape[0], X.shape[0] // Y.shape[0]))

            # Update the history            
            self.points_hist.append(np.copy(X))
            
        # self.points_hist = np.array(self.points_hist)
        # self.batch_hist = np.array(self.batch_hist)
        
        return X
                
   