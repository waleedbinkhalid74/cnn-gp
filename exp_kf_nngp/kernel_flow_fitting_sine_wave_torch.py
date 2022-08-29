import jax.numpy as np
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import neural_tangents as nt
from neural_tangents import stax
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')

from KernelFlow import KernelFlowsNP
from KernelFlow import KernelFlowsNPJAX

def train_network(model, dataloader, epochs, learning_rate, val_set = False):
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    criterion = nn.MSELoss()
    total_loss = []
    last_loss = 100
    trigger_times = 0
    patience = 5

    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad() # To not accumulate gradients
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().numpy())

        # Early stoppping based on validation dataset
        if val_set:
            with torch.no_grad():
                predict_nn = model(val_set[0])
            current_loss = criterion(predict_nn.numpy(), val_set[1].numpy())
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    return model, total_loss
            else:
                trigger_times = 0
            last_loss = current_loss

    return model, total_loss
    
class SineWave(Dataset):

    def __init__(self, n=100, low_x=0.0, high_x=2*np.pi, random_train = False):
        xbound_low = low_x
        xbound_high = high_x
        target_fn = lambda x: np.sin(x)
        if random_train == True:
            x = random.sample(range(low_x, high_x), n)
            x = np.reshape(x, (x.shape[0], -1))
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

def get_torch_nn(n=1):
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        nn.Linear(32, 32), 
        nn.ReLU()
        ]

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        *layers,
        nn.Linear(32, 1),
    )
    return model

def get_network(n=1):
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        stax.Dense(32, W_std=1.5, b_std=0.05), 
        stax.Relu()
        ]

    init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return init_fn, apply_fn, kernel_fn

if __name__ == "__main__":

    low_x = 0.0
    high_x = 2*np.pi
    train_points = 10
    test_points = 100

    train_dataset = SineWave(train_points, low_x=low_x, high_x=high_x)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size = int(train_points), shuffle=False)
    train_xs, train_ys = next(iter(train_dataloader))

    test_dataset = SineWave(test_points)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size = test_points, shuffle=False)
    test_xs, test_ys = next(iter(test_dataloader))

    fig, ax = plt.subplots(1, 1)

    network_depth_arr = [2, 4, 8, 14]
    for n in tqdm(network_depth_arr):
        network_depth = n

        learning_rate = 0.001
        training_steps = 10000

        model = get_torch_nn(n)
        model, total_loss = train_network(model=model, dataloader=train_dataloader, epochs=training_steps, learning_rate=learning_rate)
        with torch.no_grad():
            predict_nn = model(test_xs)

        ax.plot(test_xs, predict_nn, linewidth=2, label=f"""Depth = {network_depth}""", alpha=0.75)

    init_fn, apply_fn, kernel_fn = get_network(1)

    # Evaluation with Kernel Flow with NNGP Kernel 
    KF_JAX = KernelFlowsNPJAX(kernel_fn, regularization_lambda=0.001)
    iterations = 1000
    batch_size = train_points
    _ = KF_JAX.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01)
    prediction_KF_JAX = KF_JAX.predict(test_xs)

    # Evaluation with Kernel Flow with RBF Kernel
    mu = np.array([2.0])
    kernel_name = "RBF"
    KF_RBF = KernelFlowsNP(kernel_name, mu)
    X_train_perturbed_rbf = KF_RBF.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01, reg=0.00001)
    predict_kf_rbf = KF_RBF.predict(test_xs)

    ax.plot(test_xs, prediction_KF_JAX, '--', linewidth=2, label='Kernel Flow - NNGP')
    ax.plot(test_xs, predict_kf_rbf, '--', linewidth=2, label='Kernel Flow - RBF')
    ax.plot(train_xs, train_ys, 'o', label="Train")
    ax.plot(test_xs, test_ys, '--', linewidth=2, label='Test')
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    plt.legend()
    fig.savefig('figs/dummy.png')