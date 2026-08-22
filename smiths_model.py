import numpy as np
import matplotlib.pyplot as plt

def generate_rayleigh_fading_smith(
    f_c: float, freq: np.ndarray, v_kph: float, f_s: float, N: int
):
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



if __name__ == "__main__":
    from utils import *
    from config import CONFIG
    from qpsk_modulation import modulate_qpsk
    bit_length = CONFIG["bit_length"]
    pilot_spacing = CONFIG["pilot_spacing"]
    f_s = CONFIG["f_s"]
    f_c = CONFIG["f_c"]
    speeds = CONFIG["speeds"]
    Eb_N0_dB_range = CONFIG["Eb_N0_dB_range"]
    model = CONFIG["model"]
    bit_stream = generate_bits(bit_length)
    qpsk_sig = modulate_qpsk(bit_stream)
    N = len(qpsk_sig)
    freq = np.fft.fftfreq(N, d=1 / f_s)
    plt.figure(figsize=(10, 6))
    freq_shifted = np.fft.fftshift(freq)
    max_fd_overall = 0
    for v_kph in speeds:
        h_t, fd_max, S = generate_rayleigh_fading_smith(f_c, freq, v_kph, f_s, N)
        S_f_mag = np.fft.fftshift(S)
        if fd_max > max_fd_overall:
            max_fd_overall = fd_max
        plt.plot(
            freq_shifted,
            S_f_mag,
            label= f"{v_kph} km/h ($f_{{d,{{max}}}}$ = {fd_max:.1f} Hz)",
            alpha=0.75,
        )
    plt.xlim(-max_fd_overall * 1.2, max_fd_overall * 1.2)

    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude $|S(f)|$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig('doppler_spectrum.pdf')
    plt.show()

