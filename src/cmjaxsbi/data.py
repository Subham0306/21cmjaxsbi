import os
import numpy as np

def sample_pk(k, Pk, k_bins):
    return np.interp(k_bins, k, Pk)

def preprocess_X(X_clean, mean=None, std=None):
    X = np.clip(X_clean, 1e-6, None)
    X = np.log10(X)
    if mean is not None:
        X = (X - mean) / std
    X = np.clip(X, -8.0, 8.0)
    return X

def add_ska_noise_catalog(X_clean, sigma_total):
    X2d = X_clean.reshape(-1, 8, 8)
    noise = np.random.normal(
        loc=0.0,
        scale=sigma_total,
        size=X2d.shape
    )
    X_noisy = X2d + noise
    X_noisy = np.maximum(
        X_noisy,
        1e-6
    )
    return X_noisy.reshape(-1, 64)