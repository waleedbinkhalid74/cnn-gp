from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import os 

def test_kernel_evaluation():
    """Sanity check to ensure if reproducability exists upon changes made to source code.
    """
    model = Sequential(Conv2d(kernel_size=3),
                        ReLU(),
                        Conv2d(kernel_size=3, stride=2),
                        ReLU(),
                        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
                    )
    
    X_test = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_input.pt")
    K_xx = model(X_test, X_test) 
    K_xx_compare = torch.load(os.getcwd() + "/test/cnn_gp/data/test_kernel_output.pt")    
    assert torch.equal(K_xx, K_xx_compare)
    
    