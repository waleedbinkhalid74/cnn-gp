import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Reference: https://discuss.pytorch.org/t/why-different-results-when-multiplying-in-cpu-than-in-gpu/1356/2
# Reference: https://stackoverflow.com/questions/70650236/why-does-pytorch-matmul-get-different-results-when-executed-on-cpu-and-gpu

# torch.manual_seed(123)


X = torch.rand((2000,2000))
Y = torch.rand((2000,2000))

data_size_arr = np.arange(100, 2100, 100)
normed_diff = []
for data_size in tqdm(data_size_arr):
    x = X[:data_size, :data_size]
    y = Y[:data_size, :data_size]
    cpu_matmul = torch.matmul(x,y)
    cpu_matmul_inv = torch.linalg.inv(cpu_matmul)

    gpu_matmul = torch.matmul(x.cuda(),y.cuda())
    gpu_matmul_inv = torch.linalg.inv(gpu_matmul)

    # np_matmul = np.matmul(x.numpy(), y.numpy())
    # np_matmul_inv = np.linalg.inv(np_matmul)

    normed_diff.append(torch.linalg.norm(cpu_matmul_inv - gpu_matmul_inv.cpu()))

fig, ax = plt.subplots(1,1)

ax.bar(data_size_arr, normed_diff, width=100)
ax.set_xlabel("Square matrix height")
ax.set_ylabel("$||(X*X)^{-1}_{CPU} - (X*X)^{-1}_{GPU}||_F$")
plt.show()
plt.tight_layout()
fig.savefig('./figs/cpu_vs_gpu_matmul_inv_norm.png')