import numpy as np
from scipy import signal

def qammod(data, M, mapping='gray', inputtype='bit', unitaveragepow=False):
    if M == 4 and inputtype == 'bit':
        if len(data) % 2 != 0:
            data = np.append(data, 0) 
            
        symbols = np.zeros(len(data)//2, dtype=complex)
        
        for i in range(0, len(data), 2):
            if i+1 < len(data):
                bit_pair = data[i:i+2]
                if np.array_equal(bit_pair, [0, 0]):
                    symbols[i//2] = 1 + 1j
                elif np.array_equal(bit_pair, [0, 1]):
                    symbols[i//2] = 1 - 1j
                elif np.array_equal(bit_pair, [1, 0]):
                    symbols[i//2] = -1 + 1j
                else:  # [1, 1]
                    symbols[i//2] = -1 - 1j
        
        if unitaveragepow:
            symbols = symbols / np.sqrt(2)
            
        return symbols
    else:
        raise NotImplementedError(f"QAM modulation for M={M} not implemented")

def qamdemod(symbols, M, mapping='gray', outputtype='bit', unitaveragepow=False):
    if M == 4 and outputtype == 'bit':  # QPSK
        if unitaveragepow:
            symbols = symbols * np.sqrt(2)
            
        num_symbols = len(symbols)
        bits = np.zeros(num_symbols * 2, dtype=int)
        
        for i in range(num_symbols):
            symbol = symbols[i]
            if symbol.real >= 0 and symbol.imag >= 0:
                bits[2*i:2*i+2] = [0, 0]
            elif symbol.real >= 0 and symbol.imag < 0:
                bits[2*i:2*i+2] = [0, 1]
            elif symbol.real < 0 and symbol.imag >= 0:
                bits[2*i:2*i+2] = [1, 0]
            else:
                bits[2*i:2*i+2] = [1, 1]
        
        return bits
    else:
        raise NotImplementedError(f"QAM demodulation for M={M} not implemented")

def awgn(signal, snr, measured='measured'):
    if measured == 'measured':
        sig_power = np.mean(np.abs(signal)**2)
        noise_power = sig_power / (10**(snr/10))
    else:
        noise_power = 1 / (10**(snr/10))

    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) / np.sqrt(2)
    noisy_signal = signal + np.sqrt(noise_power) * noise
    return noisy_signal
