import torch
import torch.nn as nn


class RBF_Kernel(nn.Module):
    def __init__(self, parameters) -> None:
        super().__init__()
        sigma = nn.Parameter(torch.Tensor([parameters]))
        self.register_parameter('sigma', sigma)


    def norm_matrix(self, matrix_1, matrix_2):
        norm_square_1 = torch.sum(torch.square(matrix_1), axis = 1)
        norm_square_1 = torch.reshape(norm_square_1, (-1,1))

        norm_square_2 = torch.sum(torch.square(matrix_2), axis = 1)
        norm_square_2 = torch.reshape(norm_square_2, (-1,1))

        inner_matrix = torch.matmul(matrix_1, torch.transpose(matrix_2, 0, 1))

        norm_diff = -2 * inner_matrix + norm_square_1 + torch.transpose(norm_square_2, 0, 1)

        return norm_diff

    def __call__(self, matrix_1, matrix_2, same = None, diag= None):
        matrix = self.norm_matrix(matrix_1, matrix_2)
        K =  torch.exp(-matrix/ (2* self.sigma**2))
        return K

