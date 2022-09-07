
import numpy as np
from tqdm import tqdm
from scipy import integrate
import torch
from scipy.interpolate import griddata
from .auxilary_functions import *
#%%

class KernelFlowsNP_ODE():
    
    def __init__(self, kernel_keyword: str, parameters: list):
        """Class constructor for ODE based Non parametric kernel regression

        Args:
            kernel_keyword (str): keyword for the choice of kernel
            parameters (list): parameters that form the kernel.
        """
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        self.regularization_lambda = 0.0
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.rho = []
        self.points_hist = []
        self.batch_hist = []
        self.perturbation = []
        self.X_norm = []
        # Variables needed for flow transform --> This is a rough implementation geared towards experimentation
        self.X_batch = []
        self.Y_batch = []
        self.sample_indices = []
        self.batch_indices = []

    def G(self, t: list, X: np.ndarray, Y: np.ndarray, batch_indices: np.ndarray, sample_indices: np.ndarray, not_batch) -> np.ndarray:
        """A callable function that calculates the perturbation using the Frechet derivative of rho. The function can be used as a callable in the ODE solver provided by python.

        Args:
            t (list): Tiem interval to integrate the ODE on
            X (np.ndarray): X dataset
            Y (np.ndarray): Y target of dataset
            batch_indices (np.ndarray): Batch indices
            sample_indices (np.ndarray): Sample indices selected from batch randomly without replacement

        Returns:
            np.ndarray: Preturbations calculated by frechet derivative for items in batch and by interpolation for items outside of batch
        """

        X = np.reshape(X, (Y.shape[0], X.shape[0] // Y.shape[0]))
        X_norm = np.linalg.norm(X)
        # print(X_norm)
        self.X_norm.append(X_norm)
        X_batch = X[batch_indices]
        Y_batch = Y[batch_indices]
        
        g, rho = frechet(self.parameters, X_batch, Y_batch, sample_indices, kernel_keyword = self.kernel_keyword)#, regu_lambda=self.regularization_lambda)
        # print(np.linalg.norm(g))
        self.g.append(g)
        if rho >1 or rho <0:
            print ("Rho outside allowed bounds", rho.item())
        # g_interpolate, coeff = kernel_regression(X_batch, X[not_batch], g, self.parameters, self.kernel_keyword, regu_lambda = self.regularization_lambda)
        g_interpolate, coeff = kernel_regression(X_batch, X, g, self.parameters, self.kernel_keyword, regu_lambda = self.regularization_lambda)

        # TODO: Del later
        # g_interpolate, coeff = kernel_regression(X_batch, X, g, [2.0], self.kernel_keyword, regu_lambda = self.regularization_lambda)
        self.g_large_kernel.append(g_interpolate)
        
        # g_interpolate_small_kernel, _ = kernel_regression(X_batch, X, g, [0.5], self.kernel_keyword, regu_lambda = self.regularization_lambda)
        # self.g_small_kernel.append(g_interpolate_small_kernel)

        A = np.vstack([X_batch.T, np.ones(len(X_batch))]).T
        m_1 = np.linalg.lstsq(A, g, rcond=1e-8)[0]
        g_linear = np.matmul(np.vstack([X.T, np.ones(len(X))]).T, m_1)
        self.g_linear.append(g_linear)
        # TODO: Del later
        # perturbation = np.zeros(X.shape)
        # perturbation[batch_indices] = g
        # perturbation[not_batch] = g_interpolate
        perturbation = g_interpolate#g_interpolate_small_kernel
        self.perturbation.append(np.copy(perturbation))

        self.rho_values.append(rho.item())
        return perturbation.ravel()

    def G_flow_transform(self, t: float, X_test_batch: np.ndarray, Y_batch: np.ndarray, sample_indices: np.ndarray, test_size: int) -> np.ndarray:
        """A callable function that calculates the perturbation using the Frechet derivative of rho. 
        The function can be used as a callable in the ODE solver provided by python specifically to transform points for which
        y targets do not exist.

        Args:
            t (float): Time
            X_test_batch (np.ndarray): Combined array of test points and the batch points selected during the training of the kernel flows algorithm
            Y_batch (np.ndarray): Targets of the batches used during training 
            sample_indices (np.ndarray): Sample indices choosen for the batch during the training
            test_size (int): Size of test set

        Returns:
            np.ndarray: Perturbations based on interpolation from the batch set.
        """
        X_batch = X_test_batch[test_size:]
        X_test_batch = np.reshape(X_test_batch, (X_test_batch.shape[0]//(X_batch.shape[0] // Y_batch.shape[0]), X_batch.shape[0] // Y_batch.shape[0]))        
        X_batch = np.reshape(X_batch, (Y_batch.shape[0], X_batch.shape[0] // Y_batch.shape[0]))

        g, rho = frechet(self.parameters, X_batch, Y_batch, sample_indices, kernel_keyword = self.kernel_keyword)  
        if rho >1 or rho <0:
            print ("Rho outside allowed bounds", rho.item())
        g_interpolate, _ = kernel_regression(X_batch, X_test_batch, g, self.parameters, self.kernel_keyword, regu_lambda = self.regularization_lambda)
        # TODO: Del later
        # g_interpolate, _ = kernel_regression(X_batch, X_test_batch, g, [4.0], self.kernel_keyword, regu_lambda = self.regularization_lambda)

        A = np.vstack([X_batch.T, np.ones(len(X_batch))]).T
        m_1 = np.linalg.lstsq(A, g, rcond=1e-8)[0]
        g_linear = np.matmul(np.vstack([X_test_batch.T, np.ones(len(X_test_batch))]).T, m_1)
        self.g_linear.append(g_linear)
        # TODO: Del later

        perturbations = g_linear#.ravel()

        return perturbations.ravel()
    
    def fit(self, X: np.ndarray, Y: np.ndarray, iterations: int, batch_size: int, reg: float = 0.000001) -> np.ndarray:
        """Fit method to optimize a given kernel using non-parametric kernel flows using an ODE solver step for the updating of the datapoints

        Args:
            X (np.ndarray): Training dataset
            Y (np.ndarray): Training dataset targets
            iterations (int): Number of iterations to execute 
            batch_size (int): Batch size in each iteration to be choosen randomly without replacement
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
        self.debug_g_diff_norm = []
        self.g = []
        self.g_small_kernel = []
        self.g_large_kernel = []
        
        self.g_linear = []
        self.g_linear_2 = []
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
            self.batch_indices.append(np.copy(batch_indices))
            # The indices of all the elements not in the batch
            not_batch = np.setdiff1d(data_set_ind, batch_indices)
            solution = integrate.solve_ivp(self.G, [0, 1.0],  X.ravel(), args=(Y, batch_indices, sample_indices, not_batch), method='RK45')#, max_step=0.01) # Also tried Radau
            X = solution.y[:,-1]
            self.rho_values[-solution.nfev:]
            self.rho.append(np.sum(self.rho_values[-solution.nfev:]) / solution.nfev)
            X = np.reshape(X, (Y.shape[0], X.shape[0] // Y.shape[0]))

            # Update the history            
            self.points_hist.append(np.copy(X))
        
        return X
                
    def flow_transform(self, X_test: np.ndarray) -> np.ndarray:
        """Transforms the test set based on the the training batches.

        Args:
            X_test (np.ndarray): Points to be perturbed

        Returns:
            np.ndarray: Perturbed test points
        """

        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
            X_test = np.copy(X_test)

        test_size = 1
        for test_shape in X_test.shape:
            test_size*=test_shape    
        # test_size = X_test.shape[0]
        X_test_batch = np.concatenate((X_test, self.X_batch[0])).ravel()
        for i in tqdm(range(self.iteration)):
            X_test_batch[test_size:] = self.X_batch[i].ravel()
            # X_test_batch = X_test_batch#.ravel()
            solution = integrate.solve_ivp(self.G_flow_transform, [0, 1.0],  X_test_batch.ravel(), args=(self.Y_batch[i], self.sample_indices[i], test_size), method='RK45')
            X_test_batch = solution.y[:,-1]
            X_test_batch = X_test_batch#.reshape(-1,1)
        return X_test_batch[:test_size].reshape(X_test.shape[0], X_test.shape[1])#, X_test_batch[test_size:]

    def predict(self, X_test: np.ndarray, regu_lambda: float = 1e-4) -> np.ndarray:
        """Predict the test points based on kernel ridge regression with non-parametric ODE based kernel flows
        to improve the kernel 

        Args:
            X_test (np.ndarray): Test points to predict

        Returns:
            np.ndarray: Predicted targets of test points.
        """
        # Transforming using the flow
        test_transformed = self.flow_transform(X_test)  
        # Fetching the train set transformed
        X_train = self.points_hist[-1]
        Y_train = self.Y_train
        prediction, _ = kernel_regression(X_train, test_transformed, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu_lambda) 
        return prediction
    
    def predict_train(self):
        X_train = self.X
        Y_train = self.Y_train
        prediction, coeff = kernel_regression(X_train, X_train, Y_train, self.parameters, self.kernel_keyword, regu_lambda = self.regularization_lambda) 

        return prediction

