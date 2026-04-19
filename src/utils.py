# src/utils.py

import numpy as np

def clip_values(series, min_val, max_val):
    return np.clip(series, min_val, max_val)

def add_noise(size, low=-0.05, high=0.05):
    return np.random.uniform(low, high, size)