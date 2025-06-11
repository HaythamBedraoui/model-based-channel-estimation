import numpy as np
import os
import scipy.io as sio
from domain_change import domain_change
from encapsulate import encapsulate
from interpolate import interpolate
from PreProc import PreProc
from functions import qammod, qamdemod, awgn

class RayleighChannel:
    def __init__(self, sample_rate, max_doppler_shift, path_delays, average_path_gains, normalize_path_gains=False, path_gains_output_port=False):
        self.sample_rate = sample_rate
        self.max_doppler_shift = max_doppler_shift
        self.path_delays = path_delays
        self.average_path_gains = average_path_gains
        self.normalize_path_gains = normalize_path_gains
        self.path_gains_output_port = path_gains_output_port
        self.path_gains = None
        
    def reset(self):
        np.random.seed(None)
        
    def __call__(self, signal_in):
        signal_in = np.array(signal_in).flatten()
        path_gains_lin = 10**(np.array(self.average_path_gains)/20)
        output = np.zeros(len(signal_in), dtype=complex)
        h_ideal = np.zeros(len(self.path_delays), dtype=complex)
        
        for i, (delay, gain) in enumerate(zip(self.path_delays, path_gains_lin)):
            h_complex = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * gain
            h_ideal[i] = h_complex
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(signal_in):
                output[delay_samples:] += h_complex * signal_in[:len(signal_in)-delay_samples]
        
        return output, h_ideal

# Sim parameters
Nfft = 2048
GI = 1/8
Ng = int(Nfft * GI)
Nofdm = Nfft + Ng
Nsym = 10000
Nps = 8  # Pilot spacing, Numbers of pilots and data per OFDM symbol
Np = Nfft//Nps
Nd = Nfft-Np
Nbps = 2
M = 2**Nbps  # Number of bits per (modulated) symbol
mm = 1  # LS channel estimation

# Channel Generation
fs = 10e6

# Loopback channel
DopplerShift = 10

NUM = 1
if NUM == 1:
    Delay_samples = np.array([10, 22])
    Attenuation = np.array([0, -20, -25])
elif NUM == 2:
    Delay_samples = np.array([0.3, 4, 6, 8, 20, 22, 24, 27, 30])
    Attenuation = np.array([0, -13, -25, -23, -31, -26, -31, -25, -33, -23])
elif NUM == 3:
    Delay_samples = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.98, 1.86, 6.34, 6.78]) * 10
    Attenuation = np.array([0, -6.3, -8.8, -11.1, -8, -8.2, -6, -10.9, -12.4, -11, -16.3, -10.8])

Attenuation_lin = 10**(Attenuation/20)
k = 1/np.sqrt(np.sum(Attenuation_lin**2))  # Energy normalization
Attenuation_lin = Attenuation_lin*k
Attenuation = 20*np.log10(Attenuation_lin)

rayChan = RayleighChannel(
    sample_rate=fs, 
    max_doppler_shift=DopplerShift,
    path_delays=np.concatenate([[0], Delay_samples/fs]),
    average_path_gains=Attenuation,
    normalize_path_gains=False,
    path_gains_output_port=True
)
rayChan.reset()

# Transmitted signal generation and channel convolution
XLong = np.zeros((Nfft, Nsym), dtype=complex)
HLong = np.zeros((Nfft, Nsym), dtype=complex)
YLong = np.zeros((Nfft, Nsym), dtype=complex)
HLSLong = np.zeros((Nfft, Nsym), dtype=complex)
MSE = np.zeros(Nsym)
SNR = 10
noise = 0

for nsym in range(Nsym):
    np.random.seed(nsym+1)
    Xp = 2*(np.random.randn(Np)>0) - 1  # Pilot sequence generation

    np.random.seed(nsym+1)
    msgint = np.random.randint(0, 2, Nbps * (Nfft - Np)) # bit generation
    
    # Modulate
    Data = qammod(msgint, M, mapping='Gray', inputtype='bit', unitaveragepow=True)
    
    X = np.zeros(Nfft, dtype=complex)
    ip = 0
    pilot_loc = []
    
    for k in range(Nfft):
        if (k % Nps) == 0:
            X[k] = Xp[k // Nps]
            pilot_loc.append(k)
            ip += 1
        else:
            X[k] = Data[k-ip]
    XLong[:, nsym] = X
    
    x = np.fft.ifft(X, Nfft)   # IFFT
    xt = np.concatenate((x[Nfft-Ng:], x))  # Add CP
    
    # Channel convolution
    y_channel, hh_ideal = rayChan(xt)
    
    # Retrieve the ideal channel
    h = np.zeros(Nfft, dtype=complex)
    path_delays = np.concatenate([[0], Delay_samples/fs])
    hh_ideal_delay = np.round(path_delays * fs).astype(int)
    
    for idx_path in range(len(hh_ideal_delay)):
        if hh_ideal_delay[idx_path] < len(h):
            h[hh_ideal_delay[idx_path]] = hh_ideal[idx_path]
    
    sig_pow = np.mean(np.abs(y_channel)**2)
    H = np.fft.fft(h, Nfft)
    HLong[:, nsym] = H
    
    channel_length = len(h)  # True channel and its time-domain length
    H_power_dB = 10 * np.log10(np.abs(H)**2) # True channel power in dB
    
    # Noise
    np.random.seed(1)
    yt = awgn(y_channel, SNR, measured='measured')
    
    # Receiver
    y = yt[Ng:Nofdm]  # Remove CP
    Y = np.fft.fft(y)  # FFT
    YLong[:, nsym] = Y
    
    if mm == 1:  # LS estimation
        LS_est = np.zeros(Np, dtype=complex)
        for k in range(Np):
            LS_est[k] = Y[pilot_loc[k]] / Xp[k]
        H_est = interpolate(LS_est, np.array(pilot_loc) + 1, Nfft, 'linear')
        # Low pass frequency filtering
        h_est = np.fft.ifft(H_est)
        h_filt = np.zeros(len(h_est), dtype=complex)
        h_filt[:channel_length] = h_est[:channel_length]
        H_filt = np.fft.fft(h_filt)
        H_est_power_dB = 10 * np.log10(np.abs(H_filt)**2)
    else:
        # AI Channel estimation
        pass
    
    HLSLong[:, nsym] = H_filt
    MSE[nsym] = np.mean(np.abs(H - H_filt)**2)
    
    # Equalization
    Y_eq = Y / H_filt
    
    # Remove the pilot carriers
    ip = 0
    Data_extracted = np.zeros(Nd, dtype=complex)
    data_idx = 0
    
    for k in range(Nfft):
        if k in pilot_loc:
            ip += 1
        else:
            Data_extracted[data_idx] = Y_eq[k]
            data_idx += 1
    
    # Demodulation
    msg_detected = qamdemod(Data_extracted, M, mapping='Gray', outputtype='bit', unitaveragepow=True)
    noiseVar = 10**(-SNR/10)
    
    if (nsym + 1) % 500 == 0:
        print(f'Simulated symbols = {nsym + 1}')

BER = noise / (len(msgint) * Nsym)

dataWindow = 14
labelWindow = 14
slim = True
channel_size = 2048

hEncNoise_long, hEnc_long = PreProc(HLSLong, HLong, dataWindow, labelWindow, slim, channel_size)
hEncNoise = hEncNoise_long[:int(Nfft * GI), :, :]
hEnc = hEnc_long[:int(Nfft * GI), :, :]

python_output_dir = '../python_output/'

# Create the directory
os.makedirs(python_output_dir, exist_ok=True)

np.save(os.path.join(python_output_dir, 'h_perfect.npy'), hEnc)
np.save(os.path.join(python_output_dir, 'h_ls_estimation.npy'), hEncNoise)