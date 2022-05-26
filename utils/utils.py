import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class tqdm_skopt(object):

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res=1):
        self._bar.update(1)

def get_MNIST_dataset(train_size: int, val_size: int, shuffle: bool = True, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('MNIST_dataset/train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('MNIST_dataset/val', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size, shuffle=shuffle)

    dataiter = iter(trainloader)
    X_train, Y_train = dataiter.next()
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    Y_train = F.one_hot(Y_train, 10).to(torch.float32)

    dataiter_val = iter(valloader)
    X_test, Y_test = dataiter_val.next()
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
