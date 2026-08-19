from demodulate import demodulate_qpsk_mmse
from demodulate import demodulate_qpsk_zf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from jakes_spectrum import generate_rayleigh_fading_smith
from demodulate import *

# Number of bits to generate, 10^5 bits generated
# with a seed value for reproducibility purposes.
bit_size = int(1e5)
def generate_bits(bit_size):
    #np.random.seed(42)
    bits = np.random.randint(2, size=(bit_size))
    return bits

def modulate_qpsk(bits):
    bit_size = len(bits)
    # Reshaped the continuous bitstream to pairs of bits for qpsk modulation scheme
    # and separate the bits to get the real and imaginary parts and normalize by 1/sqrt(2) to
    # have average symbol energy of 1.
    bit_reshape = np.reshape(bits, (int(bit_size / 2), 2))
    # print(bit_reshape)

    b1 = bit_reshape[:, 0]
    b0 = bit_reshape[:, 1]
    # print(b1)
    # print(b0)
    real_part = 2 * b1 - 1
    imag_part = 2 * b0 - 1
    qpsk_signal = np.sqrt(1 / 2) * (real_part + 1j * imag_part)
    # print(qpsk_signal)
    return qpsk_signal

def insert_pilot(qpsk_sig, pilot_spacing):
    N = len(qpsk_sig)
    pilot_indices = np.arange(0, N, pilot_spacing)
    pilot_sym = (1 + 1j)/np.sqrt(2)
    tx_sig = qpsk_sig.copy()
    for i in pilot_indices:
        tx_sig[i] = pilot_sym
    return tx_sig, pilot_indices, pilot_sym

def gen_awgn_noise(length, Eb_N0_dB):
    Eb_N0 = 10 ** (Eb_N0_dB / 10)
    N_0 = 1 / (2 * Eb_N0)
    noise_r = np.random.randn(length) * np.sqrt(N_0 / 2)
    noise_i = np.random.randn(length) * np.sqrt(N_0 / 2)
    noise_sig = noise_r + 1j * noise_i
    return noise_sig

def estimate_channel(rx_symbol, pilot_indices, pilot_symbol, N):
    h_pilot = rx_symbol[pilot_indices]/pilot_symbol
    time_pilot = pilot_indices
    time_all = np.arange(N)
    real_interp = interp1d(time_pilot, h_pilot.real, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    imag_interp = interp1d(time_pilot, h_pilot.imag, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    h_est = real_interp(time_all) + 1j * imag_interp(time_all)
    return h_est


def calculate_ber(detected_bits, bits, pilot_indices=None):
    ## BER calculation
    if pilot_indices is not None:
        # Each pilot SYMBOL corresponds to 2 BITS
        pilot_bit_indices = np.concatenate([pilot_indices*2, pilot_indices*2 + 1])
        # Create a boolean mask: True for data bits, False for pilot bits
        mask = np.ones(len(bits), dtype=bool)
        mask[pilot_bit_indices] = False
        
        # Keep only data bits
        detected_bits = detected_bits[mask]
        bits = bits[mask]
    num_errors = np.sum(detected_bits != bits)
    ber = num_errors / len(bits)
    return ber

def simulate_qpsk():
    bit_size = int(1e5)
    bits = generate_bits(bit_size)
    qpsk_signal = modulate_qpsk(bits)

    pilot_spacing = 15
    tx_sig, pilot_indices, pilot_sym = insert_pilot(qpsk_signal, pilot_spacing)
    N = len(qpsk_signal)
    f_s = 10e3
    f_c = 2e9
    freq = np.fft.fftfreq(int(N), d=1/f_s)
    
    speeds = [30, 60, 120]
    Eb_N0_dB_range = np.arange(0, 21, 2)

    BER_results = {}

    for v_kph in speeds:
        h_t, fd_max, s = generate_rayleigh_fading_smith(f_c, freq, v_kph, f_s, N)
        BER_array = np.zeros(len(Eb_N0_dB_range))

        for i, Eb_N0_dB in enumerate(Eb_N0_dB_range):
            Eb_N0_linear = 10**(Eb_N0_dB/10)
            N0 = 1/(2*Eb_N0_linear)
            noise_sig = gen_awgn_noise(len(qpsk_signal), Eb_N0_dB)
            received_sig = h_t * tx_sig + noise_sig
            h_est = estimate_channel(received_sig, pilot_indices, pilot_sym, N)
            detected_bits = demodulate_qpsk_zf(received_sig, h_t)
            ber = calculate_ber(detected_bits, bits, pilot_indices)
            BER_array[i] = ber

        BER_results[v_kph] = BER_array
        print(f"Done: {v_kph} kmph, fd_max={fd_max:.2f} Hz")

    return Eb_N0_dB_range, BER_results

Eb_N0_dB_range, BER_results = simulate_qpsk()

plt.figure()
for v_kph, BER_array in BER_results.items():
    plt.semilogy(Eb_N0_dB_range, BER_array, marker='o', label=f'{v_kph} kmph')

gamma_lin = 10**(Eb_N0_dB_range/10)
theory_ber = 0.5*(1 - np.sqrt(gamma_lin/(1+gamma_lin)))
plt.semilogy(Eb_N0_dB_range, theory_ber, 'k--', label='Theoretical Rayleigh')

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR under Rayleigh Fading (Smith model)")
plt.legend()
plt.grid(True, which='both')
plt.show()
    

   

# if __name__ == "__main__":
#     simulate_qpsk()