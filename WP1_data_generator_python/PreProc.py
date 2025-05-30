import numpy as np
from domain_change import domain_change
from encapsulate import encapsulate

def PreProc(HWide, HWideLOOPnorm, dataWindow, labelWindow, slim, chan_size):
    x = domain_change(HWide, True)
    y = domain_change(HWideLOOPnorm, True)
    
    x = x[:chan_size, :]
    y = y[:chan_size, :]
    
    if slim:
        xEnc = encapsulate(x, dataWindow)
        yEnc = encapsulate(y, labelWindow)
    else:
        xEnc,yEnc = encapsulate(x , y, labelWindow)

    hEncNoise = xEnc
    hEnc = yEnc
    
    return hEncNoise, hEnc
