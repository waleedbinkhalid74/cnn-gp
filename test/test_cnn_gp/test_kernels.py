from re import S
import torch.nn.functional as F
from cnn_gp.kernels import ReLUCNNGP
from torch.autograd import gradcheck
import torch

# def test_kernel_evaluation():
#     """Sanity check to ensure if reproducability exists upon changes made to source code.
#     """
#     model = Sequential(Conv2d(kernel_size=3),
#                         ReLU(),
#                         Conv2d(kernel_size=3, stride=2),
#                         ReLU(),
#                         Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
#                     )

#     X_test = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_input.pt")
#     K_xx = model(X_test, X_test)
#     K_xx_compare = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_output.pt")
#     assert torch.equal(K_xx, K_xx_compare)

def test_relucnngp_backprop():
    """Test if self implemented backprop of ReLU Kernel is correct
    """
    relucnngp = ReLUCNNGP.apply
    input = (torch.rand((3,3,1,1),dtype=torch.double,requires_grad=True), torch.rand((3,3,1,1),dtype=torch.double,requires_grad=True))
    test = gradcheck(relucnngp, input, eps=1e-10, atol=1e-4)
    assert test

def test_predict():
    """Test if the prediction is done correctly
    """
    pass





