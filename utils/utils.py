import os
from typing import Union
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

class tqdm_skopt(object):

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res=1):
        self._bar.update(1)

def get_dataset(dataset:str, train_size: int, val_size: int, shuffle: bool = True, device: str = 'cpu')-> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the dataset specified by the user

    Args:
        dataset (str): name of the dataset --> 'cifar', 'mnist'
        train_size (int): Training size
        val_size (int): Validation size
        shuffle (bool, optional): Flag to shuffle the data. Defaults to True.
        device (str, optional): If the data is to be ported to the GPU or should it remain on the CPU. Defaults to 'cpu'.

    Returns:
        Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Traing dataset, Train dataset targets, Test dataset, Test dataset targets
    """
    if dataset == 'mnist':
        return get_MNIST_dataset(train_size=train_size, val_size=val_size, shuffle=shuffle, device=device)
    elif dataset == 'cifar':
        return get_CIFAR_dataset(train_size=train_size, val_size=val_size, shuffle=shuffle, device=device)


def get_CIFAR_dataset(train_size: int, val_size: int, shuffle: bool = True, device: str = 'cpu') -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the CIFAR dataset of images and labels with a training set and a validation set

    Args:
        train_size (int): Size of training dataset
        val_size (int): Size of validation dataset
        shuffle (bool, optional): Shuffled dataset. Defaults to True.
        device (str, optional): cpu or cuda. Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        Train dataset, Train dataset targets as one hot encoding
        Test dataset, Test dataset targets
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='CIFAR_dataset/train', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='CIFAR_dataset/val', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=shuffle)

    dataiter = iter(trainloader)
    X_train, Y_train = next(dataiter)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)

    dataiter_val = iter(valloader)
    X_test, Y_test = next(dataiter_val)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    return X_train, Y_train, X_test, Y_test


def get_MNIST_dataset(train_size: int, val_size: int, shuffle: bool = True, device: str = 'cpu') -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the MNIST dataset of images and labels with a training set and a validation set

    Args:
        train_size (int): Size of training dataset
        val_size (int): Size of validation dataset
        shuffle (bool, optional): Shuffled dataset. Defaults to True.
        device (str, optional): cpu or cuda. Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        Train dataset, Train dataset targets as one hot encoding
        Test dataset, Test dataset targets
    """
    transform = transforms.Compose([transforms.ToTensor()])#,
                                    # transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=shuffle)

    dataiter = iter(trainloader)
    X_train, Y_train = next(dataiter)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)

    dataiter_val = iter(valloader)
    X_test, Y_test = next(dataiter_val)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    return X_train, Y_train, X_test, Y_test

def get_label_from_probability(prediction_probability: torch.Tensor) -> torch.Tensor:
    """Returns the argmax of the probability of each sample

    Args:
        prediction_probability (torch.Tensor): Probablity of each class for each sample of size N x d where N is the number of samples and d is the dimension or number of classes

    Returns:
        torch.Tensor: Enumerated class labels
    """
    prediction_labels = np.argmax(prediction_probability.cpu(), axis=1)
    return prediction_labels

def animate_flow(list_of_points_x, y, filename= "myanim"):
    fig = plt.figure()
    ax =  fig.add_subplot()

    x = list_of_points_x[0]
    scat = ax.plot(x, y, 'o')[0]
    ax.set_xlim([-4.0, 4.0]), 
    ax.set_xlabel('X', fontsize=24)
    ax.set_ylabel('Y', fontsize=24)

    def update_plot(i, x, y, scatter):
        scatter.set_data(x[i], y)

    ani = animation.FuncAnimation(fig, update_plot, len(list_of_points_x), fargs=(list_of_points_x, y, scat))
    # f = r"c://Users/xx/Desktop/animation.gif" 
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save(os.getcwd() + "/fig/" + filename + ".avi", writer=writervideo)