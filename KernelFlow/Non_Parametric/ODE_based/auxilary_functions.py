# The pi or selection matrix
import autograd.numpy as np
from autograd import value_and_grad 
import math
from tqdm import tqdm
from .kernel_functions import kernels_dic
from .nabla_functions import nabla_dic
default_lambda = 1e-5

def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

# The rho function
def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regularization=1e-5):
    kernel = kernels_dic[kernel_keyword]    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = regularization
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    # inverse_data_Y_data = np.linalg.lstsq(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]), Y_data, rcond=1e-6)[0]
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    # inverse_sample_Y_sample = np.linalg.lstsq(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]), Y_sample, rcond=1e-6)[0]
    
    top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
    bottom = np.matmul(Y_data.T, np.matmul(inverse_data, Y_data))
    # top = np.dot(Y_sample, inverse_sample_Y_sample)
    # bottom = np.dot(Y_data, inverse_data_Y_data)
    
    return 1 - top/bottom

# Computes the frechet derivative for KF (equation 6.5 of the original paper)
def frechet(parameters, X, Y, sample_indices, kernel_keyword = "RBF", regu_lambda = default_lambda):
    Y_sample = Y[sample_indices]

    pi = pi_matrix(sample_indices, (sample_indices.shape[0], X.shape[0])) 
    lambda_term = regu_lambda
    
    nabla = nabla_dic[kernel_keyword]
    # Computing the nabla matrix and the regular matrix
    derivative_matrix, batch_matrix = nabla(X, parameters)
    
    # Computing the Kernel matrix Inverses
    sample_matrix = np.matmul(pi, np.matmul(batch_matrix, np.transpose(pi)))
    sample_inv = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    batch_inv = np.linalg.inv(batch_matrix + lambda_term * np.identity(batch_matrix.shape[0]))
    # sample_inv_Y_sample = np.linalg.lstsq(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]), Y_sample, rcond=1e-6)[0]
    # batch_inv_Y = np.linalg.lstsq(batch_matrix + lambda_term * np.identity(batch_matrix.shape[0]), Y, rcond=1e-6)[0]
    
    # Computing the top and bottom terms
    top = np.matmul(np.transpose(Y_sample), np.matmul(sample_inv, Y_sample))
    bottom = np.matmul(np.transpose(Y), np.matmul(batch_inv, Y))
    # top = np.matmul(np.transpose(Y_sample), sample_inv_Y_sample)
    # bottom = np.matmul(np.transpose(Y), batch_inv_Y)
    
    # sample_inv_pi_Y = np.linalg.lstsq(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]), np.matmul(pi, Y), rcond=1e-6)[0]
    
    # Computing z_hat and y_hat (see original paper)
    Z_hat = np.matmul(np.transpose(pi), np.matmul(sample_inv, np.matmul(pi, Y)))
    Y_hat = np.matmul(batch_inv, Y)
    # Z_hat = np.matmul(np.transpose(pi), sample_inv_pi_Y)
    # Y_hat = batch_inv_Y
    #Computing rho   
    rho = 1- top/bottom
    # Computing the Frechet derivative
    K_y = np.squeeze(np.matmul(derivative_matrix, Y_hat), axis = 2)
    K_z = np.squeeze(np.matmul(derivative_matrix, Z_hat), axis = 2)
    
    g = 2*((1-rho)* Y_hat * K_y - Z_hat * K_z) 
    g = g/bottom
    return g, rho


"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices
    

# Splits the data into the target and predictor variables.
def split(data):
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    
    # The test matrix 
    t_matrix = kernel(X_test, X_train, param)
    
    # Regression coefficients in feature space
    coeff, _, _, _ = np.linalg.lstsq(k_matrix, Y_train, rcond=1e-6)
    # coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    
    prediction = np.matmul(t_matrix, coeff)
    
    return prediction, coeff

def kernel_regression_coeff(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = default_lambda):
    kernel = kernels_dic[kernel_keyword]

    # The data matrix (theta in the original paper)
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    
    # Regression coefficients in feature space
    # coeff = np.matmul(np.linalg.inv(k_matrix), Y_train)
    coeff, _, _, _ = np.linalg.lstsq(k_matrix, Y_train, rcond=1e-6)

    return coeff
    
# This function ccomputes epsilon (relative: max relative transaltion = rate, absolute: max translation = rate)
def compute_LR(rate, old_points, g_pert, type_epsilon = "relative"):
    if type_epsilon == "relative":
        norm_old = np.linalg.norm(old_points, axis = 1)
        norm_pert = np.linalg.norm(g_pert, axis = 1)


        #Replace all tiny values by 1
        norm_pert[norm_pert < 0.000001] = 1
        ratio = norm_old/norm_pert
        
        epsilon = rate * np.amin(ratio)
    elif type_epsilon == "absolute":
        norm_pert = np.linalg.norm(g_pert, axis = 1)
        norm_pert[norm_pert < 0.000001] = 1
        epsilon = rate / np.amax(norm_pert)
    
    elif type_epsilon == "usual":
        epsilon = rate
    else:
        print("Error type of epsilon")
    return epsilon
#%%

# For Autograd Implementation
def grad_rho(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", reg = 0.000001):
    grad_K = value_and_grad(rho, 1)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, regularization = reg)
    return rho_value, gradient