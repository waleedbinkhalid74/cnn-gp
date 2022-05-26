import numpy as np
from utils import get_label_from_probability
import torch

def test_get_label_from_probability():
    input = torch.tensor([[0.25, 0.36, 0.9],[0.1, 0.8, 0.2],[0.0, 0.7, 0.4]])
    labels = get_label_from_probability(input)
    labels_control = torch.tensor([2, 1, 1])
    assert torch.equal(labels, labels_control)