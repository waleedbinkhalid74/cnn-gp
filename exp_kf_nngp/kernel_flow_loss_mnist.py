from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms as T
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys

# sys.path.insert(0, os.getcwd() + '/.')
# from KernelFlow import KernelFlowsNP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
# Val_size = 1000
test_size = 10000

transform_train = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
# transform_val = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
transform_test = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])

trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform_train)
trainset = torch.utils.data.Subset(trainset, np.arange(0,60000))
# valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_val)
# valset = torch.utils.data.Subset(valset, np.arange(0,1000))
testset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_test)
testset = torch.utils.data.Subset(testset, np.arange(0,10000))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# valloader = torch.utils.data.DataLoader(valset, batch_size=Val_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=True)

def get_network(depth):
    class ResBlock(torch.nn.Module):
        def __init__(self, in_size:int, hidden_size:int):
                super().__init__()
                self.lin1 = nn.Linear(in_size, hidden_size)
                # self.lin2 = nn.Linear(hidden_size, out_size)
                self.batchnorm1 = nn.BatchNorm1d(hidden_size)
                # self.batchnorm2 = nn.BatchNorm1d(out_size)

        def linblock(self, x):
            x = F.relu(self.batchnorm1(self.lin1(x)))
            # x = F.relu(self.batchnorm2(self.lin2(x)))
            return x

        def forward(self, x): return x + self.linblock(x) # skip connection

    layers = []
    for _ in range(depth):  # n_layers
            layers += [
                    ResBlock(784,784)
                    ]

    res_model = torch.nn.Sequential(
            torch.nn.Linear(28*28, 784),
            torch.nn.BatchNorm1d(784),
            torch.nn.ReLU(),
            *layers,
            torch.nn.Linear(784, 10),
            )

    res_model.to(device)
    return res_model

def train_network(model, trainloader, epochs, learning_rate, device='cpu'):
    model.train()
    optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()
    loss_progress = []
    for epoch in range(epochs):
        for batch_id, (data, targets) in tqdm(enumerate(trainloader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            pred = model(data)
            pred = pred.squeeze()
            loss = criterion(pred, targets)
            optimizer.zero_grad() # To not accumulate gradients
            loss.backward()
            optimizer.step()
            loss_progress.append(loss.item())
         
    return loss_progress

def check_accuracy(loader, model):
    model.eval()
    no_samples = 0
    no_correct = 0
    model.eval()
    with torch.no_grad(): # Do not maintain gradient info
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            pred = model(x).cpu()
            pred = torch.argmax(pred, 1).squeeze()
            no_correct += (pred == y.cpu()).sum().numpy()
            no_samples += pred.shape[0]
    return no_correct / no_samples * 100

def get_acc_from_nn(depth_arr):
    depth_vs_test_loss = {}
    for depth in tqdm(depth_arr):
        model = get_network(depth)
        learning_rate = 0.001
        epochs = 5
        loss_progress = train_network(model, trainloader, epochs, learning_rate=learning_rate, device=device)
        accuracy = check_accuracy(testloader, model)
        depth_vs_test_loss[depth] = accuracy    
    return depth_vs_test_loss

def plot_results(training_dataset_size_arr: np.ndarray, depth_arr:np.ndarray, depth_vs_test_loss_nn: dict, batch_size_vs_loss_kf_rbf: dict):
    graph_data = {}
    fig, ax = plt.subplots(1,1)
    for i, training_size in enumerate(training_dataset_size_arr):
        graph_data[training_size] = [batch_size_vs_loss_kf_rbf[str(training_size)]]
        for key, value in depth_vs_test_loss_nn.items():
            graph_data[training_size] += [value[i]]

    # Plotting MSE vs Network Depth
    for key, value in graph_data.items():
        ax.semilogy(value, 'o-', label=f"""Number of Training points = {key}""")
    x_tick_labels = ['Kernel Flow RBF'] +  ["NN Depth = " + str(x + 1) for x in depth_arr]
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, rotation = 45)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("MSE Loss - Log scale")
    plt.legend(prop={'size': 6})
    plt.tight_layout()
    fig.savefig("figs/sine_wave_training_mse_loss_vs_arch_" + FLAGS.activation +"_torch.png")

if __name__ == "__main__":
    depth_arr = [0, 15, 30, 50]
    depth_vs_test_acc = get_acc_from_nn(depth_arr)
    lists = sorted(depth_vs_test_acc.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    fig, ax = plt.subplots(1,1)
    ax.plot(x, y)
    ax.set_ylim((0,100))
    ax.set_xlabel("Network Depth")
    ax.set_ylabel("Accuracy")
    plt.show()