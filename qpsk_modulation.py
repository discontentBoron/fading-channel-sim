import numpy as np

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

def demodulate_qpsk_zf(received_sig, h_t):
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

def demodulate_qpsk_mmse(received_sig, h_t, N0):
    h_conj = np.conj(h_t)
    mmse_s = h_conj/(np.abs(h_t)**2 + N0)
    eq_sig = mmse_s * received_sig
    detected_sig_real = eq_sig.real
    detected_sig_imag = eq_sig.imag

    detected_b1 = (detected_sig_real > 0).astype(int)
    detected_b0 = (detected_sig_imag > 0).astype(int)

    interleaved_bits = np.column_stack((detected_b1, detected_b0))
    detected_bits = interleaved_bits.flatten()
    return detected_bits