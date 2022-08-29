import torch
import torch.nn as nn


class RBF_Kernel(nn.Module):
    """This class represents the RBF kernel but adapts it as an nn.Module so that it can be trained using the automatic differenciation engine of pytorch.
    """
    def __init__(self, parameters) -> None:
        super().__init__()
        sigma = nn.Parameter(torch.Tensor([parameters]))
        self.register_parameter('sigma', sigma)


    def norm_matrix(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        """Calculates the norm difference between two matrices

        Args:
            matrix_1 (torch.Tensor): Matrix 1
            matrix_2 (torch.Tensor): Matrix 2

        Returns:
            torch.Tensor: Norm difference between matrices
        """

        norm_square_1 = torch.sum(torch.square(matrix_1), axis = 1)
        norm_square_1 = torch.reshape(norm_square_1, (-1,1))

        norm_square_2 = torch.sum(torch.square(matrix_2), axis = 1)
        norm_square_2 = torch.reshape(norm_square_2, (-1,1))

        inner_matrix = torch.matmul(matrix_1, torch.transpose(matrix_2, 0, 1))

        norm_diff = -2 * inner_matrix + norm_square_1 + torch.transpose(norm_square_2, 0, 1)

        return norm_diff

    def __call__(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor, same = None, diag= None) -> torch.Tensor:
        """Allows for calling the object instantiated by the class to calculate the kernel for two given inputs

        Args:
            matrix_1 (torch.Tensor): Matrix 1
            matrix_2 (torch.Tensor): Matrix 2
            same (_type_, optional): Ignored. Defaults to None.
            diag (_type_, optional): Ignored. Defaults to None.

        Returns:
            torch.Tensor: Evaluated kernel for the given input data
        """

        matrix = self.norm_matrix(matrix_1, matrix_2)
        K =  torch.exp(-matrix/ (2* self.sigma**2))
        return K

