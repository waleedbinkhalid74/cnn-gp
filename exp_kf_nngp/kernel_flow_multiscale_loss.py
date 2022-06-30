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
sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP

class MultiScaleSineWave(Dataset):

    def __init__(self, n=100, random=False):
        xbound_low = -np.pi
        xbound_high = np.pi#1
        target_fn = lambda x: np.sin(x) + 0.2*np.sin(10*x) + 0.1*np.sin(30*x)
        if random:
            x = xbound_low + np.random.rand(n,1) * xbound_high*2
        else:
            x = np.linspace(xbound_low, xbound_high, n)
            x = np.reshape(x, (x.shape[0], -1))
        y = target_fn(x)
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)
        self.n_samples = n

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n_samples

def get_resnet(depth):
    class ResBlock(torch.nn.Module):
        def __init__(self, in_size:int, hidden_size:int):
            super().__init__()
            self.lin1 = nn.Linear(in_size, hidden_size)
            # self.batchnorm1 = nn.BatchNorm1d(hidden_size)

        def resblock(self, x):
            # x = F.relu(self.batchnorm1(self.lin1(x)))
            x = F.relu(self.lin1(x))
            return x

        def forward(self, x): return x + self.resblock(x) # skip connection

    layers = []
    for _ in range(depth):  # n_layers
            layers += [
                    ResBlock(4,4)
                    ]

    res_model = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            # torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            *layers,
            torch.nn.Linear(4, 1),
            )
    # res_model.to(device)
    return res_model

def train_network(model, dataloader, epochs, learning_rate, val_set = False):
    model.train()
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    total_loss = []
    val_loss = []
    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            optimizer.zero_grad() # To not accumulate gradients
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().numpy())

        if val_set and epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                predict_val = model(val_set[0])
            current_loss = criterion(predict_val, val_set[1])
            print(f"""Validation loss is {current_loss}""")
            val_loss.append(current_loss)
            model.train()

    return model, total_loss, val_loss

def get_nn_loss(depth_array, axs):
    axs = axs.reshape(-1)
    depth_vs_loss = {}
    for ax in axs[:len(axs)+1]:
        ax.axis('off')

    for i, depth in tqdm(enumerate(depth_array)):
        res_model = get_resnet(depth)
        res_multiscale_trained, loss_trend, val_loss = train_network(res_model, train_loader, 20000, learning_rate=0.001, val_set=None)
        with torch.no_grad():
            res_model.eval()
            pred_y = res_model(test_x)
        mse_test = criterion(pred_y, test_y).item()
        depth_vs_loss[depth] = mse_test
        axs[i].plot(test_x, test_y, '--', label='Test Data')
        axs[i].plot(test_x, pred_y, '-', label='Test Prediction')
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"""Network Depth = {depth + 1}""")
        axs[i].legend(prop={'size': 6})
    plt.tight_layout()
    # plt.show()
    return depth_vs_loss, axs

def get_kf_loss(train_x, train_y, test_x, test_y, ax):
    mu = np.array([0.15])
    kernel_name = "RBF"
    KF_rbf = KernelFlowsNP(kernel_name, mu)
    iterations = 1
    batch_size = 100
    X_train_perturbed_rbf = KF_rbf.fit(train_x, train_y, iterations, 
                                        batch_size = batch_size, learning_rate=0.01, reg=0.0001, type_epsilon="relative")
    predict_kf_rbf = KF_rbf.predict(test_x, regu=0.0001, epsilon_choice='combination')
    criterion = torch.nn.MSELoss()
    kf_loss = criterion(torch.tensor(predict_kf_rbf), test_y)
    ax.plot(test_x, test_y, '--', label='Test Data')
    ax.plot(test_x, predict_kf_rbf, '-', label='Test Prediction')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Kernel Flow")
    ax.legend(prop={'size': 6})
    plt.show()
    return kf_loss.item()

if __name__ == '__main__':
    # Train dataset
    train_dataset = MultiScaleSineWave(100)
    train_loader = DataLoader(dataset=train_dataset, batch_size = 100, shuffle=True)
    train_x, train_y = next(iter(train_loader))
    
    # Test dataset
    test_dataset = MultiScaleSineWave(1000)
    test_loader = DataLoader(dataset=test_dataset, batch_size = 1000, shuffle=False)
    test_x, test_y = next(iter(test_loader))
    
    criterion = torch.nn.MSELoss()
    # depth_arr = [0, 1, 4, 9, 14, 19, 24, 29]
    depth_arr = [1, 9, 14, 24, 29]
    # depth_arr = [0, 1, 4, 9, 14, 19, 29, 39]
    fig, axs = plt.subplots(3, 3, figsize=(10,5))

    depth_vs_loss, axs = get_nn_loss(depth_arr, axs)
    depth_vs_loss[1000] = get_kf_loss(train_x, train_y, test_x, test_y, axs[len(depth_arr)])
    
    lists = sorted(depth_vs_loss.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    
    fig.savefig('figs/multiscale_fits.png')
    fig, ax = plt.subplots(1,1)
    ax.semilogy(y, 'o-')
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(["Depth " + str(x_i + 1) for x_i in x[:-1]] + ["Kernel Flows"], rotation=45)
    ax.set_ylabel("MSE")
    plt.tight_layout()
    plt.show()
    fig.savefig('figs/multiscale_loss.png')
