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
    freq_shifted = np.fft.fftshift(freq)
    time_axis = np.arange(N) / f_s
    max_fd_overall = 0

    fig_spec, ax_spec = plt.subplots(figsize=(10, 6))
    fig_gain, ax_gain = plt.subplots(figsize=(10, 6))

    for v_kph in speeds:
        h_t, fd_max, S = generate_rayleigh_fading_smith(f_c, freq, v_kph, f_s, N)
        S_f_mag = np.fft.fftshift(S)
        if fd_max > max_fd_overall:
            max_fd_overall = fd_max
        ax_spec.plot(
            freq_shifted,
            S_f_mag,
            label=f"{v_kph} km/h ($f_{{d,max}}$ = {fd_max:.1f} Hz)",
            alpha=0.75,
        )

        # Plot 2: Time-Domain Channel Gain |h(t)|
        ax_gain.plot(
            time_axis,
            np.abs(h_t),
            label=f"{v_kph} km/h",
            alpha=0.85,
        )
    ax_spec.set_xlim(-max_fd_overall * 1.2, max_fd_overall * 1.2)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude $|S(f)|$")
    ax_spec.set_title("Rayleigh Fading Doppler Power Spectrum")
    ax_spec.grid(True, linestyle="--", alpha=0.6)
    ax_spec.legend(loc="upper right")
    fig_spec.tight_layout()
    fig_spec.savefig("doppler_spectrum.pdf")

    ax_gain.set_xlim(0, 0.07)
    ax_gain.set_xlabel("Time (s)")
    ax_gain.set_ylabel("Channel Magnitude $|h(t)|$")
    ax_gain.set_title("Rayleigh Fading Channel Magnitude over Time")
    ax_gain.grid(True, linestyle="--", alpha=0.6)
    ax_gain.legend(loc="upper right")
    fig_gain.tight_layout()
    fig_gain.savefig("channel_gain.pdf")

    plt.show()
