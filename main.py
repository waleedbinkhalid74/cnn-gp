import jax.numpy as np

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax

from IPython.display import set_matplotlib_formats
from tqdm import tqdm
set_matplotlib_formats('pdf', 'svg')
import matplotlib
import seaborn as sns

sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

import matplotlib.pyplot as plt

def format_plot(x=None, y=None):
  # plt.grid(False)
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

legend = functools.partial(plt.legend, fontsize=10)

def plot_fn(train, test, *fs):
  train_xs, train_ys = train

  plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')

  if test != None:
    test_xs, test_ys = test
    plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

    for f in fs:
      plt.plot(test_xs, f(test_xs), '-', linewidth=3)

  plt.xlim([-np.pi, np.pi])
  plt.ylim([-1.5, 1.5])

  format_plot('$x$', '$f$')

def get_network(n=1):
    layers = []
    for _ in range(n):  # n_layers
        layers += [
        stax.Dense(512, W_std=1.5, b_std=0.05), 
        stax.Erf()
        ]

    init_fn, apply_fn, kernel_fn = stax.serial(
        *layers,
        stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return init_fn, apply_fn, kernel_fn

if __name__ == "__main__":
  key = random.PRNGKey(10)
  train_points = 5
  test_points = 50
  noise_scale = 1e-1

  target_fn = lambda x: np.sin(x)
  key, x_key, y_key = random.split(key, 3)

  train_xs = random.uniform(x_key, (train_points, 1), minval=-np.pi, maxval=np.pi)

  train_ys = target_fn(train_xs)
  train_ys += noise_scale * random.normal(y_key, (train_points, 1))
  train = (train_xs, train_ys)
  test_xs = np.linspace(-np.pi, np.pi, test_points)
  test_xs = np.reshape(test_xs, (test_points, 1))

  test_ys = target_fn(test_xs)
  test = (test_xs, test_ys)

  plt.subplot(1, 1, 1)

  for n in tqdm(range(1, 5)):
    init_fn, apply_fn, kernel_fn = get_network(n)
    key, net_key = random.split(key)
    _, params = init_fn(net_key, (-1, 1))

    learning_rate = 0.1
    training_steps = 10000

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)
    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    train_losses = []
    test_losses = []

    opt_state = opt_init(params)
    for i in range(training_steps):
      opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

      train_losses += [loss(get_params(opt_state), *train)]
      test_losses += [loss(get_params(opt_state), *test)]
    plt.plot(test_xs, apply_fn(get_params(opt_state), test_xs), linewidth=2, label=f"""Depth = {n}""")

  plot_fn(train, None)
  plt.plot(test_xs, test_ys, '--', linewidth=2, label='Test')
  plt.legend()
  finalize_plot((1.5, 0.6))
  plt.savefig('figs/increase_depthnn_fit_sine.png')