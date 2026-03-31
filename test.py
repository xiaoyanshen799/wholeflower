from transformers.trainer_utils import set_seed
import random
import numpy as np
import torch

def sample_once(seed):
    set_seed(seed)
    return {
        "random": [random.random() for _ in range(3)],
        "numpy": np.random.rand(3).tolist(),
        "torch": torch.rand(3).tolist(),
    }

a = sample_once(42)
b = sample_once(42)
c = sample_once(42)

print("seed=42 first :", a)
print("seed=42 second:", b)
print("seed=43      :", c)

