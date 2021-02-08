from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def plot_signals(s1, s2, signal, t):
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")

    ax: plt.Axes = plt.subplot(3, 1, 1)
    ax.plot(t, s1)
    ax.set_title(r"$s_1 = \sin(33Hz * 2\pi t)$")

    ax: plt.Axes = plt.subplot(3, 1, 2)
    ax.plot(t, s2)
    ax.set_title(r"$s_2 = \sin(70Hz * 2\pi t)$")

    ax: plt.Axes = plt.subplot(3, 1, 3)
    ax.plot(t, signal)
    ax.set_title(r"$s = s_1 + s_2$")

    figure: plt.Figure = ax.get_figure()
    # figure.set_size_inches((10, 10))
    figure.tight_layout(h_pad=2)
    figure.show()


def plot_spectrum(amplitudes: np.array, t: np.array, N: int):
    K = len(amplitudes)
    T = t[1] - t[0]  # sampling interval
    # The frequency resolution is determined by: df = sampling frequency / number of samples
    f = np.linspace(0, 1 / T, N)  # (0, 1000), 500 samples

    # we don't care about negative frequencies
    amplitudes = amplitudes[:K // 2]
    f = f[:K // 2]

    figure, ax = plt.subplots()
    ax.bar(f, amplitudes, width=1.5)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency [Hz]")
    figure.show()


def main():
    t = np.linspace(0, 0.5, 500)  # 1/1000 -> sampling with 1kH frequency

    # signal = sin(f * 2PIt), where f is signal frequency
    s1 = np.sin(40 * 2 * np.pi * t)
    s2 = 0.5 * np.sin(90 * 2 * np.pi * t)
    signal = s1 + s2
    plot_signals(s1, s2, signal, t)

    N = len(signal)
    K = 200

    frequencies = np.arange(0, K)
    y = dft_naive(signal, frequencies)
    amplitudes = np.abs(y) * 1 / N

    plot_spectrum(amplitudes, t, N)


if __name__ == '__main__':
    main()
