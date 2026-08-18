import numpy as np
import matplotlib.pyplot as plt

# f_c = 2e9
# f_s = 10e3
# N = int(50e3)
# t = np.arange(N) / f_s
# freq = np.fft.fftfreq(int(N), d=1 / f_s)


def generate_rayleigh_fading_smith(
    f_c: float, freq: np.ndarray, v_kph: float, f_s: float, N: int
):
    """
    Generate a complex Rayleigh fading channel using the Jakes/Smith method.

    Parameters
    ----------
    fc : float
        Carrier frequency in Hz.
    freq : np.ndarray
        Precomputed frequency axis (from np.fft.fftfreq).
    v_kph : float
        Mobile speed in km/h.
    f_s : float
        Sampling rate in Hz.
    N : int
        Number of time-domain samples to generate.
    Returns
    -------
    h_t : np.ndarray (complex)
        Time-domain fading coefficients, normalized to unit average power.
    fd_max : float
        Maximum Doppler shift in Hz.
    S : np.ndarray
        Jakes Doppler power spectrum evaluated at `freq`.
    """
    c = 3e8
    v_mps = v_kph / 3.6
    fd_max = (v_mps * f_c) / c
    mask_freqs = np.abs(freq) < fd_max
    S = np.zeros_like(freq)
    S[mask_freqs] = 1 / (np.pi * fd_max * np.sqrt(1 - (freq[mask_freqs] / fd_max) ** 2))
    noise_freq = np.random.randn(int(N)) + 1j * np.random.randn(int(N))
    H_f = noise_freq * np.sqrt(S)
    h_t = np.fft.ifft(H_f)
    h_t = h_t / np.sqrt(np.mean(np.abs(h_t) ** 2))
    return h_t, fd_max, S


# speeds = [20, 60, 120]
# results = {}
# for v in speeds:
#     h_t, fd_max, S = generate_rayleigh_fading_smith(f_c, freq, v, f_s, N)
#     results[v] = (h_t, fd_max, S)

# t = np.arange(N) / f_s

# fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# for ax, v in zip(axes, speeds):
#     h_t, fd_max, S = results[v]
#     ax.plot(t, np.abs(h_t))
#     ax.set_ylabel("|h(t)|")
#     ax.set_title(f"{v} kmph (fd_max = {fd_max:.1f} Hz)")
# axes[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()
# Plots
# Frequency domain plot of S(f)
"""S_shifted = np.fft.fftshift(S)
freq_shifted = np.fft.fftshift(freq)
plt.plot(freq_shifted, S_shifted)
plt.ylabel("S(f)")
plt.xlabel("Frequency(in Hz)")
plt.title(f"Jakes Doppler Spectrum (60 kmph, fd_max={fd_max:.1f} Hz)")
plt.show()

# Time domain plot of S(f)
plt.plot(t, np.abs(h_t))
plt.title("Time domain plot of S(f)")
plt.xlabel("Index")
plt.ylabel("s(t)")
plt.show()

# Histogram
plt.hist(np.abs(h_t))
plt.show()"""
