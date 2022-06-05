import time
from typing import Tuple
import jax.numpy as np
from jax import jit, value_and_grad , random
from tqdm import tqdm
import jax

class KernelFlowsNPJAX():

    def __init__(self, kernel, regularization_lambda: float = 0.00001):
        self.kernel = kernel
        self.regularization_lambda = regularization_lambda
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.coeff = []
        self.points_hist = []
        self.batch_hist = []
        self.epsilon = []
        self.perturbation = []

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
            batch_indices = KernelFlowsNPJAX.sample_selection(dataset_size, batch_size, batch_key)
            data_batch = len(batch_indices)
        else:
            batch_size = int(batch_size)
            batch_indices = KernelFlowsNPJAX.sample_selection(dataset_size, batch_size, batch_key)
            data_batch = len(batch_indices)

        # Sample from the mini-batch
        sample_size = int(np.ceil(data_batch*sample_proportion))
        sample_indices = KernelFlowsNPJAX.sample_selection(data_batch, sample_size, sample_key)

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
            pi = pi.at[i, sample_indices[i]].set(1)
            # pi[i][sample_indices[i]] = 1

        return pi

    def rho(self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, sample_indices: jax.numpy.ndarray) -> float:
        """Calculates the loss rho for a particular batch and sample selection from the dataset based on the priciple
            that even if the dataset is halved, the accuracy remains stable

        Args:
            X (jax.numpy.ndarray): batch dataset
            Y (jax.numpy.ndarray): targets of the batch dataset
            sample_indices (jax.numpy.ndarray): sample selected from the batch dataset

        Returns:
            float: rho value for current dataset
        """
        kernel_matrix = self.kernel(X, X, 'nngp')
        pi = KernelFlowsNPJAX.pi_matrix(sample_indices, (sample_indices.shape[0], X.shape[0]))   
        sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
        
        Y_sample = Y[sample_indices]
        
        inverse_data = np.linalg.inv(kernel_matrix + self.regularization_lambda * np.identity(kernel_matrix.shape[0]))
        inverse_sample = np.linalg.inv(sample_matrix + self.regularization_lambda * np.identity(sample_matrix.shape[0]))
    

        top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
        bottom = np.matmul(Y.T, np.matmul(inverse_data, Y))
        rho_val = 1 - np.trace(top)/np.trace(bottom)
        return rho_val

    def kernel_regression_coeff(self, X_train: jax.numpy.ndarray, Y_train: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Calculates the kernel regression coefficient K(X_train,X_train)^-1 @ Y_train

        Args:
            X_train (jax.numpy.ndarray): Training dataset
            Y_train (jax.numpy.ndarray): Training dataset targets

        Returns:
            jax.numpy.ndarray: Coefficients for kernel regression
        """
        # The data matrix (theta in the original paper)
        k_matrix = self.kernel(X_train, X_train, 'nngp')
        k_matrix += self.regularization_lambda * np.identity(k_matrix.shape[0])
        
        # Regression coefficients in feature space
        coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
        
        return coeff

    # Generate a prediction
    def kernel_regression(self, X_train: jax.numpy.ndarray, X_test: jax.numpy.ndarray, Y_train: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Performs kernel regression using K(X_test, X_train) @ K(X_train, X_train)^-1 @ Y_train

        Args:
            X_train (jax.numpy.ndarray): Training dataset
            X_test (jax.numpy.ndarray): Testing dataset
            Y_train (jax.numpy.ndarray): Training dataset targets

        Returns:
            jax.numpy.ndarray: Predicted Testing dataset targets
        """


        # The data matrix (theta in the original paper)
        k_matrix = self.kernel(X_train, X_train, 'nngp')
        k_matrix += self.regularization_lambda * np.identity(k_matrix.shape[0])
        
        # The test matrix 
        t_matrix = self.kernel(X_test, X_train, 'nngp')
        
        # Regression coefficients in feature space
        # coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
        coeff, _, _, _ = np.linalg.lstsq(k_matrix, Y_train, rcond=1e-8)        
        prediction = np.matmul(t_matrix, coeff)
        
        return prediction, coeff
        
        
    # This function ccomputes epsilon (relative: max relative transaltion = rate, absolute: max translation = rate)
    @staticmethod
    def compute_LR(rate: float, old_points: jax.numpy.ndarray, g_pert: jax.numpy.ndarray, type_epsilon: str = "relative") -> float:
        """Calculates the epsilon based on which the perturbations are added to the training dataset

        Args:
            rate (float): Original learning rate
            old_points (jax.numpy.ndarray): Batch Dataset points of the current iteration
            g_pert (jax.numpy.ndarray): Amount of perturbation for Batch Dataset points
            type_epsilon (str, optional): Strategy based on which the perturbation should be calculated. Defaults to "relative".

        Returns:
            float: Epsilon value to scale the perturbation
        """
        if type_epsilon == "relative":
            norm_old = np.linalg.norm(old_points, axis = 1)
            norm_pert = np.linalg.norm(g_pert, axis = 1)


            #Replace all tiny values by 1
            norm_pert = norm_pert.at[norm_pert < 0.000001].set(1)
            # norm_pert[norm_pert < 0.000001] = 1
            ratio = norm_old/norm_pert
            
            epsilon = rate * np.amin(ratio)
        elif type_epsilon == "absolute":
            norm_pert = np.linalg.norm(g_pert, axis = 1)
            # norm_pert[norm_pert < 0.000001] = 1
            norm_pert = norm_pert.at[norm_pert < 0.000001].set(1)
            epsilon = rate / np.amax(norm_pert)
        
        elif type_epsilon == "usual":
            epsilon = rate
        else:
            print("Error type of epsilon")
        return epsilon
        
    # def adjust_LR(iteration, LR, parameters):
    #     if parameters["adjust_type"] == False:
    #         return parameters["LR"]
    #     else:
    #         if parameters["adjust_type"] == "linear":
    #             return LR-parameters["rate"]
    #         elif parameters["adjust_type"] == "threshold":
    #             if iteration in parameters["LR_threshold"][0]:
    #                 return parameters["LR_threshold"][1][np.where(parameters["LR_threshold"][0]==iteration)[0][0]]
    #             else:
    #                 return LR
                    
    def fit(self, X: jax.numpy.ndarray, Y: jax.numpy.ndarray, iterations: int, batch_size: int = False, 
            sample_proportion: float = 0.5, learning_rate: float = 0.01, type_epsilon: str = "relative", 
            record_hist: bool = True) -> jax.numpy.ndarray:
        """Fits by perturbing the trianing dataset using the Non-parametric Kernel Flow algorithm and a NNGP kernel

        Args:
            X (jax.numpy.ndarray): Training dataset
            Y (jax.numpy.ndarray): Training dataset targets
            iterations (int): Number of iterations of the non-parameteric kernel flow algorithm to apply
            batch_size (int, optional): Batch size for the algorithm. Defaults to False.
            sample_proportion (float, optional): Proportion of the batchsize to use in the sample dataset. Defaults to 0.5.
            learning_rate (float, optional): Learning rate of the algorithm. Defaults to 0.01.
            type_epsilon (str, optional): How to calculate the proportion in which to add perturbations to the dataset. Defaults to "relative".
            record_hist (bool, optional): Flag if the history of the points should be recorded. Defaults to True.

        Returns:
            jax.numpy.ndarray: Perturbed points
        """
        # Create a copy of the parameters (so the original parameters aren't modified)
        # self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        self.iteration = iterations
        self.points_hist.append(np.copy(X))
        self.learning_rate = learning_rate
        self.type_epsilon = type_epsilon

        if batch_size == False:
            self.regression = False
        
        data_set_ind = np.arange(X.shape[0])
        perturbation = np.zeros(X.shape)
        for i in tqdm(range(iterations)):
            # Create a batch and a sample
            sample_indices, batch_indices = KernelFlowsNPJAX.batch_creation(X.shape[0], batch_size, sample_proportion=sample_proportion)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            self.batch_hist.append(np.copy(X_batch))

            # The indices of all the elements not in the batch
            not_batch = np.setdiff1d(data_set_ind, batch_indices)

            # Compute the gradient
            drho_dx = jit(value_and_grad(self.rho))
            rho, g = drho_dx(X_batch, Y_batch, sample_indices)
            g = -g
            if rho >1.0001 or rho <-0.00001:
                print ("Rho outside allowed bounds", rho)
            # Compute the perturbations by interpolation
            if batch_size == False:
                perturbation = g
                coeff = self.kernel_regression_coeff(X_batch, g)

            else:
                g_interpolate, coeff = self.kernel_regression(X_batch, X[not_batch], g)  
                perturbation = perturbation.at[batch_indices].set(g)
                # perturbation[batch_indices] = g
                perturbation = perturbation.at[not_batch].set(g_interpolate)
                # perturbation[not_batch] = g_interpolate

            #Find epsilon
            epsilon = KernelFlowsNPJAX.compute_LR(learning_rate, X, perturbation, type_epsilon = type_epsilon)
            # Adjust the learning rate based on the learning parameters
            # learning_rate = adjust_LR(i, learning_rate, learning_parameters)
            # self.LR.append(learning_rate)

            #Update the points
            X += epsilon * perturbation
            
            # Recording the regression coefficients
            self.coeff.append(coeff)
            # Update the history
            self.rho_values.append(rho)
            self.epsilon.append(epsilon)
            self.perturbation.append(perturbation)
            self.batch_hist.append(np.copy(X_batch))
            
            if record_hist == True:
                self.points_hist.append(np.copy(X))

        self.X_train = np.copy(X)
        return X

    def flow_transform(self, X_test: jax.numpy.ndarray, iterations: int, epsilon_choice: str = "combination") -> jax.numpy.ndarray:
        """Transforms the test dataset based on the perturbation history

        Args:
            X_test (jax.numpy.ndarray): Test dataset points
            iterations (int): Number of iterations to run
            epsilon_choice (str, optional): Choice of epsilon calculating strategy to scale perturbations. Defaults to "combination".

        Returns:
            jax.numpy.ndarray: Perturbed test dataset points
        """
        # Keeping track of the perturbations
        self.test_history = []
        self.test_history.append(X_test)

        # First case: mini-batch was used, hence the regression coefficients have already been computed
        for i in range(iterations):                            
            # Fetching the regression coefficients, the batch and the learning rate
            coeff = self.coeff[i]
            X_batch = self.batch_hist[i]
            learning_rate = self.learning_rate
            
                
            # Computing the regression matrix
            test_matrix= self.kernel(X_test, X_batch, 'nngp')
            
            # Prediction and perturbation
            perturbation = np.dot(test_matrix, coeff)
            
            if epsilon_choice == "historic":
                epsilon = self.epsilon[i]
            elif epsilon_choice == "new":
                epsilon = KernelFlowsNPJAX.compute_LR(learning_rate, X_test, perturbation, type_epsilon = self.type_epsilon)
            elif epsilon_choice == "combination":
                epsilon_1 = self.epsilon[i]
                epsilon_2 = KernelFlowsNPJAX.compute_LR(learning_rate, X_test, perturbation, type_epsilon = self.type_epsilon)
                epsilon = min(epsilon_1, epsilon_2)
            else:
                print("Error epsilon type ")
            X_test += epsilon * perturbation
                
            # Updating the test history
            self.test_history.append(X_test)
        return X_test

    def predict(self, X_test: jax.numpy.ndarray, kernel_it: int = -1, epsilon_choice:str = "combination") -> jax.numpy.ndarray:
        """Perturbs the test points based on the training history and then performs kernel regression on those perturbed points using 
            the already perturbed training dataset.

        Args:
            X_test (jax.numpy.ndarray): Test dataset
            kernel_it (int, optional): Number of iterations to perform on the test dataset. Defaults to -1.
            epsilon_choice (str, optional): Strategy to calculate the epsilon based on which the perturbations are scaled. Defaults to "combination".

        Returns:
            jax.numpy.ndarray: Predicted targets for the test dataset
        """
        # Transforming using the flow
        if kernel_it > 0:
           flow_test = self.flow_transform(X_test, kernel_it, epsilon_choice = epsilon_choice)  
        elif kernel_it == -1:
            flow_test = self.flow_transform(X_test, self.iteration, epsilon_choice = epsilon_choice) 
        else: 
            print("Error, Kernel iteration not understood")
            
        # Fetching the train set transformed
        if kernel_it == -1: 
            X_train = self.X_train
        elif kernel_it >0:
            X_train = self.points_hist[kernel_it]
        else:
            print("Error kernel it")
            
        Y_train = self.Y_train
            
        prediction, coeff = self.kernel_regression(X_train, flow_test, Y_train)

        return prediction