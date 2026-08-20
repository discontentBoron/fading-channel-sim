import numpy as np
CONFIG = {
    "bit_length": int(10e5),
    "pilot_spacing": 20,
    "f_s": 10e3,
    "f_c": 2e9,
    "model": 'jk', # 'jk' for Jake's method, 'zx' for Zheng-Xiao's method
    "speeds": [20, 60, 120],
    "Eb_N0_dB_range": np.arange(0, 26, 2),
    "M_zx": 12,    # only for Zheng-Xiao
    "model_name" : {"jk":"Smith's Model", "zx": "Zheng-Xiao's Model"}
}