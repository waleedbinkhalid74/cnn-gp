import jax.numpy as jnp
import numpy as np
from jax import jit
from neural_tangents import stax
from jax import random
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch
import torch.nn.functional as F

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

def get_flat_mnist():
    batch_size = 50000
    Val_size = 1000
    test_size = 1000

    transform_train = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
    transform_val = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])
    transform_test = transforms.Compose([transforms.ToTensor(), T.Lambda(lambda x: torch.flatten(x)), T.Lambda(lambda x:  x / (torch.linalg.norm(x)))])

    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform_train)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_val)
    testset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=Val_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=True)

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    X_train = X_train.numpy()
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)
    Y_train = Y_train.numpy()

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()
    return X_train, Y_train, X_test, Y_test

def get_kernel(depth, W_std, b_std, activation = stax.Relu(), out_dims = 10):
    # W_std = np.random.rand()*100
    # b_std = np.random.rand()*100
    layers = []
    for i in range(depth - 1):
        layers += [stax.Dense(25, W_std=W_std, b_std=b_std), activation]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers,
                stax.Dense(out_dims, W_std=W_std, b_std=b_std))
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return kernel_fn

def get_training_test_points(n_train, n_test, rand_key, low_x = 0.0, high_x = 2*np.pi):
    train_points = n_train
    test_points = n_test
    noise_scale = 0*1e-2

    target_fn = lambda x: np.sin(x)

    # train_xs = random.uniform(x_key, (train_points, 1), minval=low_x, maxval=high_x)
    train_xs = np.linspace(low_x, high_x, train_points)
    train_xs = np.reshape(train_xs, (train_points, 1))

    train_ys = target_fn(train_xs)

    # key, x_key, y_key = random.split(key, 3)
    train_ys += noise_scale * random.normal(rand_key, (train_points, 1))
    train = (train_xs, train_ys)
    test_xs = np.linspace(low_x, high_x, test_points)
    test_xs = np.reshape(test_xs, (test_points, 1))

    test_ys = target_fn(test_xs)
    test = (test_xs, test_ys)
    return train, test