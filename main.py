from utils import generate_channel
from qpsk_modulation import modulate_qpsk, demodulate_qpsk_zf
from utils import *
from config import CONFIG
import matplotlib.pyplot as plt


def run_sim():
    bit_length = CONFIG["bit_length"]
    pilot_spacing = CONFIG["pilot_spacing"]
    f_s = CONFIG["f_s"]
    f_c = CONFIG["f_c"]
    speeds = CONFIG["speeds"]
    Eb_N0_dB_range = CONFIG["Eb_N0_dB_range"]
    model = CONFIG["model"]

    # Generate random bitstream
    bit_stream = generate_bits(bit_length)
    qpsk_sig = modulate_qpsk(bit_stream)

    # Insert pilot bits into qpsk_sig
    tx_sig, pilot_indices, pilot_sym = insert_pilot(qpsk_sig, pilot_spacing)
    N = len(qpsk_sig)
    BER_results = {}
    for v_kph in speeds:
        h_t, fd_max = generate_channel(model, N, f_s, f_c, v_kph)
        BER_array = np.zeros(len(Eb_N0_dB_range))

        for i, Eb_N0_dB in enumerate(Eb_N0_dB_range):
            noise_sig = gen_awgn_noise(len(qpsk_sig), Eb_N0_dB)
            received_sig = h_t * tx_sig + noise_sig
            h_est = estimate_channel(received_sig, pilot_indices, pilot_sym, N)
            detected_bits = demodulate_qpsk_zf(received_sig, h_est)
            ber = calculate_ber(detected_bits, bit_stream, pilot_indices)
            BER_array[i] = ber

        BER_results[v_kph] = BER_array
        print(f"Done: {v_kph} kmph, fd_max={fd_max:.2f} Hz")

    return Eb_N0_dB_range, BER_results

Eb_N0_dB_range, BER_results = run_sim()

plt.figure()
for v_kph, BER_array in BER_results.items():
    plt.semilogy(Eb_N0_dB_range, BER_array, marker='o', label=f'{v_kph} kmph')

gamma_lin = 10**(Eb_N0_dB_range/10)
theory_ber = 0.5*(1 - np.sqrt(gamma_lin/(1+gamma_lin)))
plt.semilogy(Eb_N0_dB_range, theory_ber, 'k--', label='Theoretical Rayleigh')

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.title(f"BER vs SNR under Rayleigh Fading using {CONFIG["model_name"][CONFIG["model"]]} ")
plt.legend()
plt.grid(True, which='both')
plt.show()
