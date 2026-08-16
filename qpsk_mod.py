import numpy as np
# import matplotlib.pyplot as plt

# Number of bits to generate, 10^5 bits generated
# with a seed value for reproducibility purposes.
bit_size = int(1e5)
np.random.seed(42)
bits = np.random.randint(2, size=(bit_size))
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
# Generate the noise signal and distort the transmitted QPSK
# signal for different values of normalized(?) SNR.
Eb_N0_dB = np.arange(0, 21, 2)
BER_array = np.zeros(len(Eb_N0_dB))
for i, val in enumerate(Eb_N0_dB):
    Eb_N0 = 10 ** (val / 10)
    N_0 = 1 / (2 * Eb_N0)
    noise_r = np.random.randn(len(qpsk_signal)) * np.sqrt(N_0 / 2)
    noise_i = np.random.randn(len(qpsk_signal)) * np.sqrt(N_0 / 2)
    noise_sig = noise_r + 1j * noise_i
    received_sig = qpsk_signal + noise_sig
    print("----------------------------------------------------------")
    print(f"Eb_N0 = {val}")
    print(f"QPSK = {qpsk_signal[:5]} \n")
    print(f"received_sig = {received_sig[:5]} \n")
    ## Demodulation
    detected_sig_real = received_sig.real
    detected_sig_imag = received_sig.imag
    print(f"Received signal(Re): {detected_sig_real[:5]} \n")
    print(f"Received signal(Im): {detected_sig_imag[:5]} \n")
    detected_b1 = (detected_sig_real > 0).astype(int)
    detected_b0 = (detected_sig_imag > 0).astype(int)
    print(f"Received signal bit b1: {detected_b1[:5]} \n")
    print(f"Received signal bit b0: {detected_b0[:5]} \n")
    interleaved_bits = np.column_stack((detected_b1, detected_b0))
    print(f"Interleaved bits: {interleaved_bits[:5]}")
    detected_bits = interleaved_bits.flatten()
    print(f"Decoded bits: {detected_bits[:10]}")

    ## BER calculation
    num_errors = np.sum(detected_bits != bits)
    ber = num_errors / len(bits)
    BER_array[i] = ber
    print(f"Bit Error Rate: {BER_array} \n")
