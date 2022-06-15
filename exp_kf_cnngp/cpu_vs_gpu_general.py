import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# Reference: https://discuss.pytorch.org/t/why-different-results-when-multiplying-in-cpu-than-in-gpu/1356/2
# Reference: https://stackoverflow.com/questions/70650236/why-does-pytorch-matmul-get-different-results-when-executed-on-cpu-and-gpu

torch.manual_seed(int(time.time()))


# A = torch.rand((2000,2000))
# b = torch.rand(2000)
Ni_min = 1500
Ni_max = 2100
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
    # cpu_matmul_float64 = torch.matmul(A_i.to(torch.float64), B_i.to(torch.float64))
    cpu_lstsq_float64 = torch.linalg.lstsq(A_i.to(torch.float64), b_i.to(torch.float64), rcond=1e-8).solution
    cpu_lstsq_float64 = torch.matmul(B_i.to(torch.float64), cpu_lstsq_float64)
    # cpu_kg_float64 = torch.matmul(B_i.to(torch.float64), cpu_lstsq_float64)
    # cpu_lstsq_float64 = torch.matmul(torch.linalg.inv(A_i), b_i)

    # CPU Evaluations Float 32
    # cpu_matmul_float32 = torch.matmul(A_i, A_i)
    cpu_lstsq_float32 = torch.linalg.lstsq(A_i, b_i, rcond=1e-8).solution
    cpu_lstsq_float32 = torch.matmul(B_i, cpu_lstsq_float32)
    # cpu_kg_float32 = torch.matmul(B_i, cpu_lstsq_float32)
    # cpu_lstsq_float32 = torch.matmul(torch.linalg.inv(A_i.to(torch.float32)), b_i.to(torch.float32))

    # GPU Evaluations Float 64
    # gpu_matmul_float64 = torch.matmul(A_i.to(torch.float64).cuda(), B_i.to(torch.float64).cuda())
    gpu_lstsq_float64 = torch.linalg.lstsq(A_i.to(torch.float64).cuda(), b_i.to(torch.float64).cuda(), rcond=1e-8).solution
    gpu_lstsq_float64 = torch.matmul(B_i.to(torch.float64).cuda(), gpu_lstsq_float64)
    # gpu_kg_float64 = torch.matmul(B_i.to(torch.float64).cuda(), gpu_lstsq_float64)
    # gpu_lstsq_float64 = torch.matmul(torch.linalg.inv(A_i.cuda()), b_i.cuda())

    # CPU Evaluations Float 32
    # gpu_matmul_float32 = torch.matmul(A_i.cuda(), B_i.cuda())
    gpu_lstsq_float32 = torch.linalg.lstsq(A_i.cuda(), b_i.cuda(), rcond=1e-8).solution
    gpu_lstsq_float32 = torch.matmul(B_i.cuda(), gpu_lstsq_float32)

    # gpu_kg_float32 = torch.matmul(B_i.to(torch.float32).cuda(), gpu_lstsq_float32)
    # gpu_lstsq_float32 = torch.matmul(torch.linalg.inv(A_i.to(torch.float32).cuda()), b_i.to(torch.float32).cuda())

    # Norm of CPU vs GPU on float 64
    normed_diff_float64.append(torch.linalg.norm((cpu_lstsq_float64 - gpu_lstsq_float64.cpu())))
    # normed_diff_matmul_float64.append(torch.linalg.norm((cpu_matmul_float64 - gpu_matmul_float64.cpu()) / cpu_matmul_float64))
    # normed_diff_kerreg_float64.append(torch.linalg.norm(cpu_kg_float64 - gpu_kg_float64.cpu()))
    
    # Norm of CPU vs GPU on float 32
    normed_diff_float32.append(torch.linalg.norm((cpu_lstsq_float64 - gpu_lstsq_float32.cpu())))
    # normed_diff_matmul_float32.append(torch.linalg.norm((cpu_matmul_float64 - gpu_matmul_float32.cpu()) / cpu_matmul_float64))
    # normed_diff_kerreg_float32.append(torch.linalg.norm(cpu_kg_float32 - gpu_kg_float32.cpu()))

    # Norm of CPU float 64 vs float 32
    normed_diff_float32_vs_float64.append(torch.linalg.norm((cpu_lstsq_float32 - cpu_lstsq_float64)))
    # normed_diff_matmul_float32_vs_float64.append(torch.linalg.norm((cpu_matmul_float32 - cpu_matmul_float64) / cpu_matmul_float32))
    # normed_diff_kerreg_float32_vs_float64.append(torch.linalg.norm(cpu_kg_float32 - cpu_kg_float64))

    normed_diff_float32_vs_float64_gpu.append(torch.linalg.norm((gpu_lstsq_float32.cpu() - gpu_lstsq_float64.cpu())))
    # normed_diff_matmul_float32_vs_float64_gpu.append(torch.linalg.norm((gpu_matmul_float32.cpu() - gpu_matmul_float64.cpu()) / gpu_matmul_float32.cpu()))

fig, ax = plt.subplots(1,1)
ax.semilogy(data_size_arr, normed_diff_float64, 'o-', label='CPU 64 bit vs GPU 64 bit precision', alpha=0.35)
ax.semilogy(data_size_arr, normed_diff_float32, '*-', label='CPU 64 bit vs GPU 32 bit precision', alpha=0.35)
ax.semilogy(data_size_arr, normed_diff_float32_vs_float64, '^-', label='CPU 32 bit vs CPU 64 bit', alpha=0.35)
ax.semilogy(data_size_arr, normed_diff_float32_vs_float64_gpu, '^-', label='GPU 32 bit vs GPU 64 bit', alpha=0.35)
ax.set_xlabel("Square matrix height")
ax.set_ylabel("$||A^{-1}_{CPU}b_{CPU} - A^{-1}_{GPU}b_{GPU}||_2$")
plt.legend()
plt.tight_layout()
plt.show()
# fig.savefig('./figs/cpu_vs_gpu_lstsq_norm.png')

# fig, ax = plt.subplots(1,1)
# ax.semilogy(data_size_arr, normed_diff_matmul_float64, 'o-', label='CPU vs GPU 64 bit precision', alpha=0.35)
# ax.semilogy(data_size_arr, normed_diff_matmul_float32, '*-',  label='CPU vs GPU 32 bit precision', alpha=0.35)
# ax.semilogy(data_size_arr, normed_diff_matmul_float32_vs_float64, '^-',  label='Float 32 vs Float 64 on CPU', alpha=0.35)
# ax.semilogy(data_size_arr, normed_diff_matmul_float32_vs_float64_gpu, '^-',  label='Float 32 vs Float 64 on GPU', alpha=0.35)
# ax.set_xlabel("Matrix size N_i")
# ax.set_ylabel("$||AA_{CPU} - AA_{GPU}||_2$")
# ax.set_xticks(data_size_arr)
# ax.set_xticklabels(ax.get_xticks(), rotation = 45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.semilogy(data_size_arr, normed_diff_kerreg_float64, 'o-', label='CPU vs GPU 64 bit precision')
# ax.semilogy(data_size_arr, normed_diff_kerreg_float32, '*-',  label='CPU vs GPU 32 bit precision')
# ax.semilogy(data_size_arr, normed_diff_kerreg_float32_vs_float64, '^-',  label='Float 32 vs Float 64 on CPU')
# ax.set_xlabel("Matrix size N_i")
# ax.set_ylabel("$||AA_{CPU} - AA_{GPU}||_2$")
# ax.set_xticks(data_size_arr)
# ax.set_xticklabels(ax.get_xticks(), rotation = 45)
# plt.legend()
# plt.tight_layout()
# plt.show()
# pass
# fig.savefig('./figs/cpu_vs_gpu_lstsq_norm.png')