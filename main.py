from cnn_gp import Sequential, Conv2d, ReLU, resnet_block
import torch 

if __name__ == "__main__":
    
    # X = torch.rand((2,1,28,28))
    # torch.save(X, "test_kernel_input.pt")
    X = torch.load("test_kernel_input.pt")
    model = Sequential(
                Conv2d(kernel_size=3),
                ReLU(),
                Conv2d(kernel_size=3, stride=2),
                ReLU(),
                Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
                )
    
    K_xx = model(X,X)
    torch.save(K_xx, 'test_kernel_output.pt')
    