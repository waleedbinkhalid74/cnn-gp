import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import absl.app
import sys
import os

sys.path.insert(0, os.getcwd() + '/.')

from utils import get_MNIST_dataset, get_label_from_probability
from cnn_gp import NNGPKernel
from KernelFlow import KernelFlowsTorch
from configs import kernel_flow_configs

FLAGS = absl.app.flags.FLAGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(_):
    X_train, Y_train, X_test, Y_test = get_MNIST_dataset(train_size=50000, val_size=1000, device=DEVICE)
    cnn_gp = kernel_flow_configs.get_CNNGP(model_name = FLAGS.CNNGP_model, device=DEVICE)


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("CNNGP_model", "covnet",
                    "which CNNGP model to test on")
    absl.app.run(main)
