from cnn_gp import Sequential, Conv2d, ReLU, NNGPKernel
import numpy as np
import time

def get_CNNGP(model_name: str = 'covnet', device: str = 'cpu')-> NNGPKernel:
    """Gets the CNN GP equivilant NN depending on the flag from the user. Defaults to Covnet

    Returns:
        NNGPKernel: CNNGP Kernel
    """
    np.random.seed(int(time.time()))
    var_weight = np.random.rand()*100.0
    var_bias = np.random.rand()*100.0    
    if model_name == 'covnet':
        print("Selected Covnet")
        layers = []
        for _ in range(7):  # n_layers
            layers += [
                Conv2d(kernel_size=7, padding="same"),
                ReLU(),
            ]
            cnn_gp = Sequential(var_weight, var_bias,
                *layers,
                Conv2d(kernel_size=28, padding=0),
                )
    elif model_name == 'simple':
        print("Selected simple 3 layer network")
        cnn_gp = Sequential(var_weight, var_bias,
                Conv2d(kernel_size=3),
                ReLU(),
                Conv2d(kernel_size=3, stride=2),
                ReLU(),
                Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
                )
    elif model_name == "alonso_etal_covnet":
        print("Selected alonso et al covnet")
        var_bias = 7.86
        var_weight = 2.79
        layers = []
        for _ in range(7):  # n_layers
            layers += [
                Conv2d(kernel_size=7, padding="same"),
                ReLU(),
            ]
            cnn_gp = Sequential(var_weight, var_bias,
                *layers,
                Conv2d(kernel_size=28, padding=0),
                )
    return cnn_gp.to(device)