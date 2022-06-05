from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.getcwd() + '/.')
from KernelFlow import KernelFlowsNP

def train_network(model, dataloader, epochs, learning_rate):
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    total_loss = []
    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad() # To not accumulate gradients
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().numpy())
    return model, total_loss

class SineWave(Dataset):

    def __init__(self, n=100, low_x=0.0, high_x=2*np.pi):
        xbound_low = low_x
        xbound_high = high_x
        target_fn = lambda x: np.sin(x)
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

def get_mse_loss(pred, y):
    return 0.5* np.mean((pred-y)**2)

def get_loss_from_nn(training_dataset_size_arr, depth_arr):
    print("Evaluating Neural Networks...")
    test_size = 100

    depth_vs_test_loss = {}
    low_x = 0.0
    high_x = 2*np.pi
    for training_size in tqdm(training_dataset_size_arr):
        print(f"""Training Size: {training_size}""")
        train_dataset = SineWave(training_size, low_x=low_x, high_x=high_x)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size = int(training_size), shuffle=False)

        test_dataset = SineWave(test_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = test_size, shuffle=False)
        test_xs, test_ys = next(iter(test_dataloader))

        for n in tqdm(depth_arr):#, 2, 3, 5, 10]):
            network_depth = n
            net = get_torch_nn(network_depth)
            # key, net_key = random.split(key)
            learning_rate = 0.001
            training_steps = 10000
            train_network(model=net, dataloader=train_dataloader, 
                            epochs=training_steps, learning_rate=learning_rate)

            # print(loss(get_params(opt_state), *test))
            with torch.no_grad():
                predict_nn = net(test_xs)
            if network_depth in depth_vs_test_loss:
                depth_vs_test_loss[network_depth].append(get_mse_loss(predict_nn.numpy(), test_ys.numpy()))
            else:
                depth_vs_test_loss[network_depth] = [get_mse_loss(predict_nn.numpy(), test_ys.numpy())]
            
    return depth_vs_test_loss

def get_loss_from_kf_rbf(training_dataset_size_arr):
    print("Evaluating RBF Kernel with Kernel Flow...")
    test_size = 100
    batch_size_vs_loss = {}
    low_x = 0.0
    high_x = 2*np.pi
    for training_size in tqdm(training_dataset_size_arr):
        print(f"""Training Size: {training_size}""")
        train_dataset = SineWave(training_size, low_x=low_x, high_x=high_x)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size = int(training_size), shuffle=False)
        train_xs, train_ys = next(iter(train_dataloader))
        train_xs = train_xs.numpy()
        train_ys = train_ys.numpy()

        test_dataset = SineWave(test_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = test_size, shuffle=False)
        test_xs, test_ys = next(iter(test_dataloader))
        test_xs = test_xs.numpy()
        test_ys = test_ys.numpy()

        mu = np.array([2.0])
        kernel_name = "RBF"
        KF_rbf = KernelFlowsNP(kernel_name, mu)
        iterations = 1000
        batch_size = training_size
        X_train_perturbed_rbf = KF_rbf.fit(train_xs, train_ys, iterations, batch_size = batch_size, learning_rate=0.01, reg=0.0001)
        predict_kf_rbf = KF_rbf.predict(test_xs, regu=0.0001)
        batch_size_vs_loss[str(training_size)] = get_mse_loss(pred=predict_kf_rbf, y=test_ys)
    return batch_size_vs_loss

if __name__ == "__main__":
    training_dataset_size_arr = np.arange(5, 11, 1, dtype=int)
    depth_arr = [0, 1, 2, 3, 4, 5, 8, 10, 13, 15]
    depth_vs_test_loss_nn = get_loss_from_nn(training_dataset_size_arr, depth_arr)
    batch_size_vs_loss_kf_rbf = get_loss_from_kf_rbf(training_dataset_size_arr)


    # Plotting MSE vs Network Depth
    graph_data = {}
    fig, ax = plt.subplots(1,1)
    for i, training_size in enumerate(training_dataset_size_arr):
        graph_data[training_size] = [batch_size_vs_loss_kf_rbf[str(training_size)]]
        for key, value in depth_vs_test_loss_nn.items():
            graph_data[training_size] += [value[i]]

    for key, value in graph_data.items():
        ax.plot(value, 'o-', label=f"""Number of Training points = {key}""")
    x_tick_labels = ['Kernel Flow RBF'] +  ["Depth = " + str(x + 1) for x in depth_arr]
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, rotation = 45)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig("figs/sine_wave_training_mse_loss_vs_arch_torch.png")


    # Plotting MSE vs Training size
    fig, ax = plt.subplots(1,1)
    for key, value in depth_vs_test_loss_nn.items():
        ax.plot(training_dataset_size_arr, depth_vs_test_loss_nn[key], 'o-', label=f""" NN Depth {str(key+1)}""")

    ax.plot(training_dataset_size_arr, batch_size_vs_loss_kf_rbf.values(), '*--', label='Kernel Flow - RBF')
    ax.set_xlabel("Training data size")
    ax.set_ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig("figs/sine_wave_training_mse_loss_torch.png")