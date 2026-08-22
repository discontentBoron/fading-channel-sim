import numpy as np
from scipy.interpolate import interp1d
from config import CONFIG
from zheng_xiao import generate_rayleigh_fading_zx
from smiths_model import generate_rayleigh_fading_smith

def generate_bits(bit_size):
    #np.random.seed(42)
    bits = np.random.randint(2, size=(bit_size))
    return bits

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

def generate_channel(model, N, f_s, f_c, v_kph):
    if model == 'zx':
        M_zx=CONFIG["M_zx"]
        if M_zx is None:
            raise ValueError("M_zx must be provided when model='zx'")
        h_t, fd_max = generate_rayleigh_fading_zx(M_zx, N, f_s, f_c, v_kph)
    elif model == 'jk':
        freq = np.fft.fftfreq(N, d=1 / f_s)
        h_t, fd_max, s = generate_rayleigh_fading_smith(f_c, freq, v_kph, f_s, N)
    else:
        raise ValueError(f"Unknown model '{model}', expected 'zx' or 'jk'")
 
    return h_t, fd_max