"""
This script demonstrates the differences that arise when performing matrix multiplication on the CPU vs on the GPU under different precision settings on pytorch.
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# Reference: https://discuss.pytorch.org/t/why-different-results-when-multiplying-in-cpu-than-in-gpu/1356/2
# Reference: https://stackoverflow.com/questions/70650236/why-does-pytorch-matmul-get-different-results-when-executed-on-cpu-and-gpu

torch.manual_seed(int(time.time()))

Ni_min = 100
Ni_max = 1100
step = 100

data_size_arr = np.arange(Ni_min, Ni_max, step)
normed_diff_float64 = []
normed_diff_matmul_float64 = []
normed_diff_kerreg_float64 = []

normed_diff_float32 = []
normed_diff_matmul_float32 = []
normed_diff_kerreg_float32 = []

normed_diff_float32_vs_float64 = []
normed_diff_matmul_float32_vs_float64 = []
normed_diff_kerreg_float32_vs_float64 = []

normed_diff_float32_vs_float64_gpu = []
normed_diff_matmul_float32_vs_float64_gpu = []

for data_size in tqdm(data_size_arr):
    A_i = torch.rand((data_size,data_size), dtype=torch.float32) #+ 1000
    B_i = torch.rand((data_size,data_size), dtype=torch.float32) #+ 1000
    b_i = torch.rand((data_size), dtype=torch.float32)

    # CPU Evaluations Float 64
    cpu_matmul_float64 = torch.matmul(A_i.to(torch.float64), B_i.to(torch.float64))

    # CPU Evaluations Float 32
    cpu_matmul_float32 = torch.matmul(A_i, B_i)

    # GPU Evaluations Float 64
    gpu_matmul_float64 = torch.matmul(A_i.to(torch.float64).cuda(), B_i.to(torch.float64).cuda())

    # CPU Evaluations Float 32
    gpu_matmul_float32 = torch.matmul(A_i.cuda(), B_i.cuda())

    # Norm of CPU vs GPU on float 64
    normed_diff_matmul_float64.append(torch.linalg.norm((cpu_matmul_float64 - gpu_matmul_float64.cpu()) / cpu_matmul_float64))
    
    # Norm of CPU vs GPU on float 32
    normed_diff_matmul_float32.append(torch.linalg.norm((cpu_matmul_float64 - gpu_matmul_float32.cpu()) / cpu_matmul_float64))

    # Norm of CPU float 64 vs float 32
    normed_diff_matmul_float32_vs_float64.append(torch.linalg.norm((cpu_matmul_float32 - cpu_matmul_float64) / cpu_matmul_float64))


fig, ax = plt.subplots(1,1)
ax.semilogy(data_size_arr, normed_diff_matmul_float64, 'o-', label='CPU 64 bit vs GPU 64 bit precision', alpha=0.35)
ax.semilogy(data_size_arr, normed_diff_matmul_float32, '*-',  label='CPU 64 bit vs GPU 32 bit precision', alpha=0.35)
ax.semilogy(data_size_arr, normed_diff_matmul_float32_vs_float64, '^-',  label='CPU 64 bit vs CPU 32 bit precision', alpha=0.35)
# ax.semilogy(data_size_arr, normed_diff_matmul_float32_vs_float64_gpu, '^-',  label='GPU 32 bit vs GPU 64 bit', alpha=0.35)
ax.set_xlabel("Matrix size N_i")
ax.set_ylabel("$||(AB_{CPU64} - AB_{other})/ AB_{CPU64}||_2$")
ax.set_xticks(data_size_arr)
ax.set_xticklabels(ax.get_xticks(), rotation = 45)
# ax.set_title(f"""Matrix with condition number {}""")
plt.legend(prop={'size': 6})
plt.tight_layout()
plt.show()

fig.savefig("figs/cpu_vs_gpu_general.png")
