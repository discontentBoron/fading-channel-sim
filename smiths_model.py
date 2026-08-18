import numpy as np
import matplotlib.pyplot as plt
from jakes_spectrum import generate_rayleigh_fading_smith
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

def gen_awgn_noise(length, Eb_N0_dB):
    Eb_N0 = 10 ** (Eb_N0_dB / 10)
    N_0 = 1 / (2 * Eb_N0)
    noise_r = np.random.randn(length) * np.sqrt(N_0 / 2)
    noise_i = np.random.randn(length) * np.sqrt(N_0 / 2)
    noise_sig = noise_r + 1j * noise_i
    return noise_sig


def demodulate_qpsk(received_sig, h_t):
    ## Demodulation
    epsilon = 1e-9 # Small change for divide by zero error
    mag = np.abs(h_t)
    phase = np.angle(h_t)
    mag_safe = np.maximum(mag, epsilon) #any value smaller than epsilon gets replaced by epsilon itself
    h_t_safe = mag_safe * np.exp(1j * phase) 
    eq_sig = received_sig/h_t_safe

    detected_sig_real = eq_sig.real
    detected_sig_imag = eq_sig.imag

    detected_b1 = (detected_sig_real > 0).astype(int)
    detected_b0 = (detected_sig_imag > 0).astype(int)

    interleaved_bits = np.column_stack((detected_b1, detected_b0))
    detected_bits = interleaved_bits.flatten()

    return detected_bits

def calculate_ber(detected_bits, bits):
    ## BER calculation
    num_errors = np.sum(detected_bits != bits)
    ber = num_errors / len(bits)
    return ber

def simulate_qpsk():
    bit_size = int(1e5)
    bits = generate_bits(bit_size)
    qpsk_signal = modulate_qpsk(bits)

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
            noise_sig = gen_awgn_noise(len(qpsk_signal), Eb_N0_dB)
            received_sig = h_t * qpsk_signal + noise_sig
            detected_bits = demodulate_qpsk(received_sig, h_t)
            ber = calculate_ber(detected_bits, bits)
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