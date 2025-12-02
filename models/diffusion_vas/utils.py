import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)       # Python's built-in random module
    np.random.seed(seed)    # NumPy's random module
    torch.manual_seed(seed) # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False