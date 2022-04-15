from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch
import numpy as np
from KernelFlow import KernelFlowsCNNGP
from KernelFlow import batch_creation

if __name__ == "__main__":

    # X = torch.rand((10,1,28,28))
    # indices = np.arange(X.shape[0])
    # sample_indices = np.sort(np.random.choice(indices, 5, replace= False))

    # print(sample_indices)
    # # torch.save(X, "test_kernel_input.pt")
    # X = torch.load("test_kernel_input.pt")
    # model = Sequential(
    #             Conv2d(kernel_size=3),
    #             ReLU(),
    #             Conv2d(kernel_size=3, stride=2),
    #             ReLU(),
    #             Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
    #             )

    # K_xx = model(X,X)
    # torch.save(K_xx, 'test_kernel_output.pt')


    X = torch.rand((10, 1, 2,2))
    dataset_size = X.shape[0]
    batch_size = dataset_size/2
    sample_proportion = 0.5
    samples, batches = KernelFlowsCNNGP.batch_creation(dataset_size=dataset_size,
                                                       batch_size=batch_size,
                                                       sample_proportion=sample_proportion)
    print(samples)
    print(batches)
    pi_matrix = KernelFlowsCNNGP.pi_matrix(sample_indices=samples,dimension=(len(samples), len(batches)))
    print(pi_matrix)