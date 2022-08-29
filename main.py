from KernelFlow import KernelFlowsTorch
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as T
from cnn_gp import Sequential, Conv2d, ReLU
from tqdm import tqdm
from scalene import scalene_profiler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# if __name__ == "__main__":
transform = transforms.Compose([transforms.ToTensor()])

batch_size = 50000
val_size = 1000
N_I = 1000

# MNIST
trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=True)

dataiter = iter(trainloader)
X_train, Y_train = dataiter.next()
X_train = X_train.to(device)
Y_train = Y_train.to(device)
Y_train = F.one_hot(Y_train, 10).to(torch.float32)

dataiter_val = iter(valloader)
X_test, Y_test = dataiter_val.next()
X_test = X_test.to(device)
Y_test = Y_test.to(device)

var_bias = 7.86
var_weight = 2.79
# Definition of the original model
layers = []
for _ in range(7):  # n_layers
    layers += [
        Conv2d(kernel_size=7, padding="same"),
        ReLU(),
    ]
covnet_model = Sequential(var_weight, var_bias,
    *layers,
    Conv2d(kernel_size=28, padding=0),
)

covnet_model.to(device)
parameter_bounds = [(1e-3, 50.0), (0.0, 50.0)]
KFCNNGP = KernelFlowsTorch(cnn_gp_kernel=covnet_model, device=device)
# scalene_profiler.start()
KFCNNGP.fit(X_train, Y_train, 2, 300)
# scalene_profiler.stop()
