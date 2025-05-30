import numpy as np

def domain_change(x,to_time):
    shape = x.shape
    y = np.zeros(shape, dtype=complex)
    if to_time:
        for i in range(shape[1]):
            Htemp = x[:,i]
            y[:,i] = np.fft.ifft(Htemp)
    else:
        for i in range(shape[1]):
            Htemp = x[:,i]
            y[:,i] = np.fft.fft(Htemp)
    return y
