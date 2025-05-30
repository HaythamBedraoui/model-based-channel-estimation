import numpy as np
from scipy.interpolate import interp1d

def interpolate(H_est,pilot_loc,Nfft,method):
    H_est = np.array(H_est)
    pilot_loc = np.array(pilot_loc)
    if pilot_loc[0] > 1:
        slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
        H_est = np.concatenate([[H_est[0] - slope * (pilot_loc[0] - 1)], H_est])
        pilot_loc = np.concatenate([[1], pilot_loc])

    if pilot_loc[-1] < Nfft :
        slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
        H_est = np.concatenate([H_est, [H_est[-1] + slope * (Nfft - pilot_loc[-1])]])
        pilot_loc = np.concatenate([pilot_loc, [Nfft]])

    if method[0].lower() == 'l':
        H_interpolated = np.interp(np.arange(Nfft), pilot_loc, H_est)
    else:
        H = interp1d(pilot_loc, H_est, kind='cubic',fill_value="extrapolate")
        H_interpolated = H(np.arange(Nfft))
    return H_interpolated