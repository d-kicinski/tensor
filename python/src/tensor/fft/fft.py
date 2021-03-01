import numpy as np


def dft_naive(x: np.array, k: np.array) -> np.array:
    """
        x: [1, N]
        k: [1, K]
    """
    N = x.size
    n = np.arange(N)  # [N]

    n = np.expand_dims(n, axis=0)  # [1, N]
    k = np.expand_dims(k, axis=-1)  # [K, 1]

    e = k * n  # [K, N]
    e = np.exp(-2 * np.pi * 1j / N * e)  # [K, N]
    return x @ e.T
