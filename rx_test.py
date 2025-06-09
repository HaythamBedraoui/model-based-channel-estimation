import numpy as np
import matplotlib.pyplot as plt
import os
from WP1_data_generator_python.functions import qammod, awgn
from rx_chain import rx_chain

# Load data
python_output_dir = 'python_output/'
h_perfect = np.load(os.path.join(python_output_dir,'h_perfect_M3.npy'))
h_ls_estimation = np.load(os.path.join(python_output_dir,'h_ls_estimation_M3.npy'))

Nfft = h_perfect.shape[0]
GI = 1/8              
Ng = int(Nfft * GI)        
Nofdm = Nfft + Ng         
Nps = 8                   
Np = Nfft//Nps 
Nd = Nfft-Np             
Nbps = 2                   
M = 2**Nbps

test_symbols = h_perfect.shape[2]
SNRs = [10,15,20,25,30]

results = []
for SNR in SNRs:
    print(f"\nSNR = {SNR} dB")
    total_bits = 0
    bits_error = 0
    
    for nsym in range(test_symbols):
        # time domain channel from data
        h_time = h_perfect[:,0,nsym]

        # time domain to frequency domain
        H_freq = np.fft.fft(h_time)
        
        # random data bits
        np.random.seed(nsym+1)
        tx_bits = np.random.randint(0, 2, Nbps * Nd)
        
        # pilot symbols BPSK
        np.random.seed(nsym+1)
        Xp = 2*(np.random.rand(Np)>0.5) - 1
        
        # Modulate data
        Data = qammod(tx_bits,M,mapping='Gray',inputtype='bit',unitaveragepow=True)
        
        # OFDM symbol
        X = np.zeros(Nfft,dtype=complex)
        ip = 0
        pilot_pos = []
        
        for k in range(Nfft):
            if (k % Nps) == 0:
                X[k] = Xp[k // Nps]
                pilot_pos.append(k)
                ip += 1
            else:
                X[k] = Data[k-ip]
        
        # channel in frequency domain
        Y = X * H_freq
        
        # Add noise
        Y_noisy = awgn(Y,SNR,measured='measured')
        
        # apply it to the RX chain
        rx_bits, ber, H_est = rx_chain(Y_noisy, Xp, pilot_pos, Nfft, M, tx_bits)
        
        total_bits += len(tx_bits)
        bits_error += np.sum(rx_bits != tx_bits)
        
        if (nsym + 1) % 500 == 0:
            print(f'Simulated symbols = {nsym + 1}')
    
    BER = bits_error / total_bits
    print(f"SNR = {SNR} dB,BER = {BER:.6f}")
    results.append((SNR, BER))
    
    # channel estimation vs true channel
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(np.abs(H_freq),'b-',label='True Channel')
    plt.plot(np.abs(H_est),'r--',label='Estimated Channel')
    plt.title(f'Channel Magnitude Response (M=3, SNR={SNR} dB)')
    plt.legend()
    plt.grid(True)
        
    plt.subplot(212)
    plt.plot(np.angle(H_freq), 'b-', label='True Channel')
    plt.plot(np.angle(H_est), 'r--', label='Estimated Channel')
    plt.title(f'Channel Phase Response (M=3, SNR={SNR} dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'channel_estimation_M3_SNR{SNR}.png')

# BER vs SNR
plt.figure(figsize=(10, 6))
snr_values = [r[0] for r in results]
ber_values = [r[1] for r in results]
plt.semilogy(snr_values, ber_values, 'o-')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR (M=3)')
plt.savefig('ber_vs_snr_M3.png')
plt.show()

print("\nSimulation completed") 