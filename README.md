# Rayleigh Fading Channel Simulation with QPSK

A Python simulation of QPSK modulation/demodulation over Rayleigh fading channels using two different channel models.

## What it does
 
- Generates Rayleigh fading channels using two independent methods:
  - Smith's spectral shaping method (Jakes spectrum + IFFT)
  - Zheng-Xiao sum-of-sinusoids method
- Simulates QPSK modulation, AWGN, and BER over a range of Eb/N0
- Compares perfect vs. pilot-based (estimated) channel knowledge
- Implements Zero-Forcing equalization
- Validates results against theoretical Rayleigh-QPSK BER


## Files
 
- `config.py`- Control simulation parameters here
- `main.py` - Main simulation file, calls other functions, run this to view results
- `zheng_xiao.py` - Zheng-Xiao's Rayleigh fading channel generator
- `smiths_model.py` - Jake's / Smith's Rayleigh fading channel generator
- `utils.py` - helper functions
- `qpsk_modulation.py` - QPSK Modulation and Demodulation functions

## Setup

### Clone the repo
 
```bash
git clone https://github.com/discontentBoron/fading-channel-sim.git
cd fading-channel-sim
```

 
### Linux / macOS
 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
 
### Windows
 
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
## Usage
 
```bash
python main.py
```
