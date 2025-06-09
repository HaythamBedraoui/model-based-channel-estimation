import numpy as np
from WP1_data_generator_python.interpolate import interpolate
from WP1_data_generator_python.functions import qamdemod

def rx_chain(Y, Xp, pilot_pos, Nfft, M, tx_bits):
    # LS estimation
    LS_est = np.zeros(len(pilot_pos),dtype=complex)
    for k in range(len(pilot_pos)):
        LS_est[k] = Y[pilot_pos[k]]/Xp[k]
    
    # Interpolate
    pilot_pos_array = np.array(pilot_pos)+1
    H_est = interpolate(LS_est,pilot_pos_array,Nfft,'linear')  
    
    # Equalization
    Y_eq = Y / H_est
    
    # Remove pilot carriers
    data_indices = np.array([i for i in range(Nfft) if i not in pilot_pos])
    Data_extracted = Y_eq[data_indices]
    
    # Demodulation
    Nbps = int(np.log2(M))
    rx_bits = qamdemod(Data_extracted,M,mapping='Gray',outputtype='bit',unitaveragepow=True)
    
    # BER
    bit_errors = np.sum(rx_bits != tx_bits)
    BER = bit_errors/len(tx_bits)
    return rx_bits,BER,H_est 