import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP

def train_network_classification(model: nn.Module, dataloader: DataLoader, epochs: int, learning_rate: float):
    """Helper method to train NN

    Args:
        model (nn.Module): network to train
        dataloader (DataLoader): Dataloader for the training dataset
        epochs (int): Number of epochs to execute training
        learning_rate (float): Learning rate used for training

    Returns:
        (model, training_loss): Trained model and its accompanying loss history
    """
    model.train()
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    total_loss = []
    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            optimizer.zero_grad() # To not accumulate gradients
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().numpy())

    return model, total_loss

class make_swissroll_cheesecake():
    """Class that defines the generation of the swissroll cheesecake dataset
    """
    def __init__(self, n: int=100, train=True):
        """Constructor of swissroll cheesecake.

        Args:
            n (int, optional): 0.35*n + (0.85*n-0.5*n) points are generated. This is because we do not wish to get a very entangled swissroll cheesecake. Defaults to 100.
        """
        points = np.linspace(1.5*np.pi, 4.5*np.pi, n//2)
        X_1 = points*np.sin(points)
        X_2 = points*np.cos(points)
        Y_1 = np.array(np.ones(X_1.shape[0]))
        X = np.concatenate((X_1.reshape(-1,1), X_2.reshape(-1,1)), axis=1)
        X_11 = -points*np.sin(points)+ 0.25
        X_22 = -points*np.cos(points) + 0.25
        Y_2 = np.array(-np.ones(X_11.shape[0]))
        X_ = np.concatenate((X_11.reshape(-1,1), X_22.reshape(-1,1)), axis=1)
        X = np.concatenate((X,X_))
        Y = np.concatenate((Y_1,Y_2)).reshape((-1,1))
        if train:
            self.X = torch.from_numpy(np.concatenate((X[:int(n*0.325),:], X[int(n*0.5):int(n*0.825),:]))).to(torch.float32)
            self.Y = torch.from_numpy(np.concatenate((Y[:int(n*0.325),:], Y[int(n*0.5):int(n*0.825),:]))).to(torch.float32)
        else:
            self.X = torch.from_numpy(np.concatenate((X[:int(n*0.32),:], X[int(n*0.5):int(n*0.82),:]))).to(torch.float32)
            self.Y = torch.from_numpy(np.concatenate((Y[:int(n*0.32),:], Y[int(n*0.5):int(n*0.82),:]))).to(torch.float32)

        self.n = self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index,:], self.Y[index])

    def __len__(self):
        return self.n

def get_resnet(depth: int) -> nn.Module:
    """Get a resnet of a particular size

    Args:
        depth (int): depth of the resnet

    Returns:
        nn.Module: the initialized resnet
    """
    class ResBlock(torch.nn.Module):
        def __init__(self, in_size:int, hidden_size:int):
            super().__init__()
            self.lin1 = nn.Linear(in_size, hidden_size)
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)

        def resblock(self, x):
            x = F.relu(self.batchnorm1(self.lin1(x)))
            # x = F.relu(self.lin1(x))
            return x

        def forward(self, x): return x + self.resblock(x) # skip connection

    layers = []
    for _ in range(depth):  # n_layers
            layers += [
                    ResBlock(4,4)
                    ]

    res_model = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            *layers,
            torch.nn.Linear(4, 1),
            )
    # res_model.to(device)
    return res_model

def evaluate_network(res_model: nn.Module, swissroll_dataloader: DataLoader, test_data_x: torch.Tensor, test_data_y: torch.Tensor, ax, depth: int):
    """
    Helper method that runs the training for the swissroll cheesecake dataset and returns the accuracy on the test dataset.
    """
    res_model_trained, loss_trend = train_network_classification(res_model, 
                                                                swissroll_dataloader, 250, 
                                                                learning_rate=0.001)

    res_model.eval()
    with torch.no_grad():
        test_pred = res_model(test_data_x)
    test_pred[test_pred<0] = -1
    test_pred[test_pred>0] = 1

    ax.axis('off')
    ax.scatter(test_data_x[:,0], test_data_x[:,1], c=test_pred)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"""Network Depth = {depth + 1}""")

    return (test_data_y.shape[0] - np.sum(np.abs(test_pred - test_data_y).numpy())) / test_data_y.shape[0] * 100

def evaluate_kf_rbf(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, ax):
    """
    Helper method that runs the non-parametric kernel flow training and then returns the test accuracy on the trained kernel
    """
    mu = np.array([2.0])
    kernel_name = "RBF"
    KF_rbf = KernelFlowsNP(kernel_name, mu)

    iterations = 30000
    batch_size = 32
    mu_pred = KF_rbf.fit(train_x, train_y, iterations, batch_size = batch_size, learning_rate = 0.01, type_epsilon = "relative", show_it=10000, reg=0.01)
    Y_test_pred = KF_rbf.predict(test_x, regu=0.000001)
    Y_test_pred[Y_test_pred < 0] = -1
    Y_test_pred[Y_test_pred > 0] = 1

    ax.axis('off')
    ax.scatter(test_x[:,0], test_x[:,1], c=Y_test_pred)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"""Kernel Flows RBF""")

    return (test_y.shape[0] - np.sum(np.abs(Y_test_pred - test_y))) / test_y.shape[0] * 100

if __name__ == '__main__':
    swissroll = make_swissroll_cheesecake(120, train=True)
    swissroll_test = make_swissroll_cheesecake(1000, train=False)
    swissroll_dataloader = DataLoader(dataset=swissroll, batch_size = 32, shuffle=True)
    swissroll_dataloader_test = DataLoader(dataset=swissroll_test, batch_size = 1000, shuffle=True)

    swissroll_rbf = make_swissroll_cheesecake(120)
    swissroll_dataloader_rbf = DataLoader(dataset=swissroll_rbf, batch_size = 120, shuffle=True)
    train_data_x, train_data_y = next(iter(swissroll_dataloader_rbf))
    test_data_x, test_data_y = next(iter(swissroll_dataloader_test))

    plt.scatter(train_data_x[:,0], train_data_x[:,1], c = train_data_y)
    plt.scatter(test_data_x[:,0], test_data_x[:,1], c = test_data_y)

    depth_arr = [1, 2, 4, 9, 14, 24, 29, 34]
    depth_vs_accuracy = {}
    fig, axs = plt.subplots(3, 3, figsize=(10,5))
    axs = axs.reshape(-1)

    for i, depth in tqdm(enumerate(depth_arr)):
        res_model = get_resnet(depth)
        depth_vs_accuracy[depth] = evaluate_network(res_model, swissroll_dataloader, test_data_x, test_data_y, axs[i], depth)


    X_kf = train_data_x.numpy()
    Y_kf = train_data_y.numpy()
    X_kf_test = test_data_x.numpy()
    Y_kf_test = test_data_y.numpy()
    depth_vs_accuracy[1000] = evaluate_kf_rbf(X_kf, Y_kf, X_kf_test, Y_kf_test, axs[-1])
    plt.tight_layout()
    fig.savefig('figs/swissroll_cheesecake_fits.png')

    lists = sorted(depth_vs_accuracy.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples    
    fig, ax = plt.subplots(1,1)
    ax.plot(y, 'o-')
    ax.set_ylim((0,100))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(["Depth " + str(x_i + 1) for x_i in x[:-1]] + ["Kernel Flows"], rotation=45)
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    # plt.show()

    fig.savefig('figs/swissroll_cheesecake_loss_vs_arch.png')
    