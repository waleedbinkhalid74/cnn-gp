import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
arr_size = np.arange(50, 1000, 50)
diff_norm_arr = []
for size in tqdm(arr_size):
    np_mat = np.random.rand(size,size)
    jnp_mat = jnp.array(np_mat)

    Y_np = np.arange(0,size)
    Y_jnp = jnp.arange(0,size)

    np_mat_lstsq, _, _, _ = np.linalg.lstsq(np_mat, Y_np, rcond=1e-6)
    jnp_mat_lstsq, _, _, _ = jnp.linalg.lstsq(jnp_mat, Y_jnp, rcond=1e-6)

    difference = np_mat_lstsq - jnp_mat_lstsq

    diff_norm_arr.append(np.linalg.norm(difference))

plt.plot(arr_size, diff_norm_arr)
plt.show()