import numpy as np

def get_accuracy(pred: np.array, target: np.array):
    acc = pred - target
    acc = np.where(np.abs(acc) > 0, 1, 0)
    acc = (1 - acc.sum()/len(acc)) * 100
    return acc