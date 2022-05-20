from KernelFlow import KernelFlowsCNNGP
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from cnn_gp import Sequential, Conv2d, ReLU
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 50000
    val_size = 1000
    N_I = 1000

    # MNIST
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    data_string = ''

    model = Sequential(np.random.rand()*5.0, np.random.rand()*5.0,
        Conv2d(kernel_size=3),
        ReLU(),
        Conv2d(kernel_size=3, stride=2),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
        )

    data_string += 'initial_parameters: ' + str(model.var_bias) + ', ' +  str(model.var_weight) + '\n'
    
    model.to(device)

    N_i_arr = np.arange(50, 200, 50)
    rand_acc = []
    for N_i in tqdm(N_i_arr):
        Y_predictions, k_mat, t_mat = KernelFlowsCNNGP.kernel_regression(X_test=X_test, X_train=X_train[:N_i], Y_train=Y_train[:N_i], kernel=model, regularization_lambda=0.0001, blocksize=250, device=device)
        Y_predictions_labels = np.argmax(Y_predictions.cpu(), axis=1)
        rand_acc.append(accuracy_score(Y_predictions_labels, Y_test.cpu().numpy()) * 100)

    # KF = KernelFlowsCNNGP(cnn_gp_kernel=model, device=device)
    # KF._fit_autograd(X=X_train, Y=Y_train, iterations=50, batch_size=450, sample_proportion = 0.5)
    KF_RBF = KernelFlowsCNNGP(model, lr=0.1, device=device)
    KF_RBF.fit(X_train, Y_train, 2000, 600, 0.5, method='finite difference')

    trained_acc = []
    for N_i in tqdm(N_i_arr):
        Y_predictions, k_mat, t_mat = KernelFlowsCNNGP.kernel_regression(X_test=X_test, X_train=X_train[:N_i], Y_train=Y_train[:N_i], kernel=model, regularization_lambda=0.0001, blocksize=250, device=device)
        Y_predictions_labels = np.argmax(Y_predictions.cpu(), axis=1)
        trained_acc.append(accuracy_score(Y_predictions_labels, Y_test.cpu().numpy()) * 100)

    data_string += 'final_parameters: ' + str(model.var_bias) + ', ' +  str(model.var_weight) + '\n'

    data_string += 'random_network_accuracy: ' + str(rand_acc) + '\n'
    data_string += 'trained_network_accuracy: ' + str(trained_acc) + '\n'
    print(data_string)
    import os
    if not os.path.exists(os.getcwd() + '/results'):
        os.makedirs(os.getcwd() +'/results')
    text_file = open(os.getcwd() + "/results/autograd_training.txt", "w")
    n = text_file.write(data_string)
    text_file.close()