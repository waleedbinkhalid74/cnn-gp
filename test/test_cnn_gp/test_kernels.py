from re import S
import torch.nn.functional as F
from cnn_gp.kernels import ReLUCNNGP
from torch.autograd import gradcheck
import torch


def test_relucnngp_backprop():
    """Test if self implemented backprop of ReLU Kernel is correct
    """
    relucnngp = ReLUCNNGP.apply
    # input = (torch.rand((3,3,1,1),dtype=torch.double,requires_grad=True), torch.rand((3,3,1,1),dtype=torch.double,requires_grad=True))
    input = (torch.rand((10,10,1,1),dtype=torch.double,requires_grad=True), torch.rand((10,1,1,1),dtype=torch.double,requires_grad=True), torch.rand((10,1,1),dtype=torch.double,requires_grad=True))
    test = gradcheck(relucnngp, input, eps=1e-10, atol=1e-3, rtol=1e-4)
    assert test

def test_predict():
    """Test if the prediction is done correctly
    """
    pass





