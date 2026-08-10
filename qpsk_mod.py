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
print(bit_reshape)

b1 = bit_reshape[:, 0]
b0 = bit_reshape[:, 1]
print(b1)
print(b0)
real_part = 2 * b1 - 1
imag_part = 2 * b0 - 1
qpsk_signal = np.sqrt(1 / 2) * (real_part + 1j * imag_part)
print(qpsk_signal)
