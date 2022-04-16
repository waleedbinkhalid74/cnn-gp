from copyreg import pickle
from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import numpy as np
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import batch_creation
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import autograd

if __name__ == "__main__":

    # X = torch.rand((10,1,28,28))
    # indices = np.arange(X.shape[0])
    # sample_indices = np.sort(np.random.choice(indices, 5, replace= False))

    # print(sample_indices)
    # # torch.save(X, "test_kernel_input.pt")
    # X = torch.load("test_kernel_input.pt")
    # model = Sequential(
    #             Conv2d(kernel_size=3),
    #             ReLU(),
    #             Conv2d(kernel_size=3, stride=2),
    #             ReLU(),
    #             Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
    #             )

    # K_xx = model(X,X)
    # torch.save(K_xx, 'test_kernel_output.pt')


    # print(samples)
    # print(batches)
    # pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=samples,dimension=(len(samples), len(batches)))
    # print(pi_matrix)


    # model = Sequential(
    #                 Conv2d(kernel_size=3, padding=0),
    #                 ReLU(),
    #                 )
    # K = KernelFlowsCNNGP(cnn_gp_kernel=model)
    # X = torch.ones((10, 1, 3,3), dtype=torch.float32)
    # for i in range(X.shape[0]):
    #     X[i] = X[i] * i

    # Y = torch.arange(0,10)
    # Y = F.one_hot(Y, 10)
    # Y = Y.to(torch.float32)

    # samples = np.array([2,3,4])
    # batches = np.array([1, 3, 4, 5, 7])


    # N_f = len(batches)
    # N_c = len(samples)
    # X_batch = X[batches]
    # X_sample = X[samples]
    # Y_batch = Y[batches]
    # Y_sample = Y[samples]

    # pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=samples, dimension=(N_c, N_f))
    # rho = K.rho(X_batch=X_batch, Y_batch=Y_batch, Y_sample=Y_sample, pi_matrix=pi_matrix)

    # ##### TEST COMPARISION #####
    # pi_comp =  np.array([[0, 0, 1, 0, 0],
    #                     [0, 0, 0, 1, 0],
    #                     [0, 0, 0, 0, 1]], dtype=float)

    # print(pi_matrix, pi_comp)
    # K_xx = model(X_batch, X_batch)
    # K_xx = K_xx.detach().numpy()
    # sample_matrix = np.matmul(pi_comp, np.matmul(K_xx, np.transpose(pi_comp)))
    # inverse_data = np.linalg.inv(K_xx + 0.0001 * np.identity(K_xx.shape[0]))
    # inverse_sample = np.linalg.inv(sample_matrix + 0.0001 * np.identity(sample_matrix.shape[0]))
    # top = np.matmul(Y_sample.T, np.matmul(inverse_sample, Y_sample))
    # bottom = np.matmul(Y_batch.T, np.matmul(inverse_data, Y_batch))
    # rho_comp = 1 - np.trace(top)/np.trace(bottom)
    # print(rho, rho_comp, np.abs(rho.detach().numpy() - rho_comp))
    # print(np.isclose(rho.detach().numpy(), rho_comp, 1e-2))

############################################################################################################################
    autograd.set_detect_anomaly(True)
    transform = transforms.Compose([transforms.ToTensor()
                              ])

    batch_size = 100
    val_size = 50

    # MNIST
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Fashion MNIST
    # trainset = datasets.FashionMNIST('Fashion_MNIST_dataset/train', download=True, train=True, transform=transform)
    # valset = datasets.FashionMNIST('Fashion_MNIST_dataset/val', download=True, train=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    Y_train = F.one_hot(Y_train, 10)

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()

    model = Sequential(
                Conv2d(kernel_size=3),
                ReLU(),
                Conv2d(kernel_size=3, stride=2),
                ReLU(),
                Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
                )

    KF_CNN_GP = KernelFlowsCNNGP(model)
    KF_CNN_GP.fit(X_train, Y_train, 3, 50, 0.5)

############################################################################################################################
    for param in model.parameters():
        print(param, param.grad)

    # K_xx = model(X_train, X_train)
    # Loss = torch.sum(K_xx)
    # print(Loss)
    # Loss.backward()

    # for param in model.parameters():
    #     print(param, param.grad)

