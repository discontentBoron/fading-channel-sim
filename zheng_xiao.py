import numpy as np
import random
import matplotlib.pyplot as plt


def generate_rayleigh_fading_zx(M:int, N:int, f_s:float, f_c:float, v_kph:float):
    c = 3e8
    v_mps = v_kph / 3.6
    fd_max = (f_c * v_mps/c)
    wd = 2*np.pi*fd_max
    t = np.arange(N) / f_s
    theta = random.uniform(-np.pi,np.pi)
    phi = random.uniform(-np.pi,np.pi)
    psi_n = np.array([random.uniform(-np.pi, np.pi) for _ in range(M)])

    n = np.arange(1, M+1,1)
    alpha_n = (2*n*np.pi - np.pi + theta) / (4 * M)

    Xc_t = (2/np.sqrt(M)) * np.sum(
        np.cos(psi_n)[:,None]*np.cos(wd * t * np.cos(alpha_n)[:,None] + phi), axis=0
        )
    Xs_t = (2/np.sqrt(M)) * np.sum(
        np.sin(psi_n)[:,None]*np.cos(wd * t * np.cos(alpha_n)[:,None] + phi), axis=0
        )

    h_t = Xc_t + 1j*Xs_t
    h_t = h_t / np.sqrt(np.mean(np.abs(h_t)**2))
    return h_t, fd_max

# M = 16
# N = int(50e3)
# f_s = 10e3
# f_c = 2e9
# v_kph = 30
# t, h_t = generate_rayleigh_fading_zx(M, N, f_s, f_c, v_kph)


# # Plots
# plt.plot(t, np.abs(h_t))
# plt.xlabel("Time (s)")
# plt.ylabel("|h(t)|")
# plt.title(f"Zheng-Xiao Sum-of-Sinusoids Fading ({v_kph} kmph, M={M})")
# plt.show()

# plt.hist(np.abs(h_t), bins=50)
# plt.title("Envelope PDF - Zheng-Xiao model")
# plt.show()