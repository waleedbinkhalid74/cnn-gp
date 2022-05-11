import torchvision

from cnn_gp import Conv2d, ReLU, Sequential, NormalizationModule

# Setting 4
# train_range = range(0, 12500)
# validation_range = range(12500, 13500)
# test_range=range(17000,19000)
# setting 3
# train_range = range(0, 25000)
# validation_range = range(25000, 27500)
# test_range = range(27500, 30000)
# setting 2
# train_range = range(0, 20000)
# validation_range = range(20000, 22500)
# test_range = range(22500, 25000)
# setting 1
train_range = range(0, 5000)
validation_range = range(6000, 7000)
test_range = range(7000, 8000)

dataset_name = "MNIST"
model_name = "ConvNet"
dataset = torchvision.datasets.MNIST
transforms = []
epochs = 0
in_channels = 1
out_channels = 10

var_bias = 7.86
var_weight = 2.79
# Definition of the original model
layers = []
for _ in range(7):  # n_layers
    layers += [
        Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7 ** 2,
               var_bias=var_bias),
        ReLU(),
    ]
initial_model = Sequential(
    *layers,
    Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
           var_bias=var_bias),
)
# Definition of the model which is supposed to be approximated and therefore makes use of the normalization layer
# after every convolution and before every ReLU
norm_layers = []
for _ in range(7):
    norm_layers += [
        Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7 ** 2,
               var_bias=var_bias),
        NormalizationModule(),
        ReLU(),
    ]
normalized_model = Sequential(
    *norm_layers,
    Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
           var_bias=var_bias),
)
