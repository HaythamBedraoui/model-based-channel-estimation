import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_interpolate_trainable(H_est, pilot_loc, Nfft, alpha, beta, gamma, method='linear'):
    if not isinstance(H_est, torch.Tensor):
        H_est = torch.tensor(H_est, dtype=torch.complex64)
    if not isinstance(pilot_loc, torch.Tensor):
        pilot_loc = torch.tensor(pilot_loc, dtype=torch.float32)
    
    if torch.min(pilot_loc) == 1:
        pilot_loc = pilot_loc - 1
    
    if pilot_loc[0] > 0:
        if len(pilot_loc) > 1:
            slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
            
            H_est = torch.cat([H_est[0:1] - slope * pilot_loc[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
        else:
            H_est = torch.cat([H_est[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
    
    if pilot_loc[-1] < Nfft - 1:
        if len(pilot_loc) > 1:
            slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
            H_est = torch.cat([H_est, H_est[-1:] + slope * (Nfft - 1 - pilot_loc[-1:])])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])
        else:
            H_est = torch.cat([H_est, H_est[-1:]])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])

    # Interpolate with trainable parameters
    H_interpolated = torch.zeros(Nfft, dtype=torch.complex64, device=H_est.device)
    
    for i in range(Nfft):
        left_idx = torch.searchsorted(pilot_loc, float(i), right=True) - 1
        left_idx = torch.clamp(left_idx, 0, len(pilot_loc) - 2)
        right_idx = left_idx + 1
        
        X0 = pilot_loc[left_idx]
        X1 = pilot_loc[right_idx]
        Y_beta = H_est[left_idx]   # Left pilot value
        Y_alpha = H_est[right_idx]  # Right pilot value
        
        # Calculate distance factor
        if X1 - X0 > 0:
            distance_factor = (i - X0) / (X1 - X0)
        else:
            distance_factor = 0.0
        
        # Y^i = alpha * Y_alpha + beta * Y_beta + gamma * distance_factor
        H_interpolated[i] = alpha * Y_alpha + beta * Y_beta + gamma * distance_factor
    
    return H_interpolated

def torch_interpolate(H_est, pilot_loc, Nfft, method='linear'):
    if not isinstance(H_est, torch.Tensor):
        H_est = torch.tensor(H_est, dtype=torch.complex64)
    if not isinstance(pilot_loc, torch.Tensor):
        pilot_loc = torch.tensor(pilot_loc, dtype=torch.float32)
    
    if torch.min(pilot_loc) == 1:
        pilot_loc = pilot_loc - 1
    
    if pilot_loc[0] > 0:
        if len(pilot_loc) > 1:
            slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
            H_est = torch.cat([H_est[0:1] - slope * pilot_loc[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
        else:
            H_est = torch.cat([H_est[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
    
    if pilot_loc[-1] < Nfft - 1:
        if len(pilot_loc) > 1:
            slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
            H_est = torch.cat([H_est, H_est[-1:] + slope * (Nfft - 1 - pilot_loc[-1:])])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])
        else:
            H_est = torch.cat([H_est, H_est[-1:]])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])

    # Interpolation
    target_indices = torch.arange(Nfft, dtype=torch.float32, device=pilot_loc.device)
    if method.lower().startswith('l'):
        H_real = torch.interp(target_indices, pilot_loc, H_est.real)
        H_imag = torch.interp(target_indices, pilot_loc, H_est.imag)
        H_interpolated = torch.complex(H_real, H_imag)
    else:
        # if non-linear cause PyTorch doesn't have cubic interpolation
        H_real = torch.interp(target_indices, pilot_loc, H_est.real)
        H_imag = torch.interp(target_indices, pilot_loc, H_est.imag)
        H_interpolated = torch.complex(H_real, H_imag)
    
    return H_interpolated

def torch_qamdemod(symbols, M, mapping='Gray', outputtype='bit', unitaveragepow=True):
    if not isinstance(symbols, torch.Tensor):
        symbols = torch.tensor(symbols, dtype=torch.complex64)
    
    if M == 4 and outputtype == 'bit':  # QPSK
        if unitaveragepow:
            symbols = symbols * torch.sqrt(torch.tensor(2.0, device=symbols.device))
            
        num_symbols = len(symbols)
        bits = torch.zeros(num_symbols * 2, dtype=torch.int32, device=symbols.device)

        real_part = symbols.real
        imag_part = symbols.imag

        bit0 = (real_part < 0).int()
        bit1 = (imag_part < 0).int()
        
        bits[0::2] = bit0
        bits[1::2] = bit1
        
        return bits
    else:
        raise NotImplementedError(f"QAM demodulation for M={M} not implemented")

class ChannelEstimator(nn.Module):
    def __init__(self, num_pilots=8):
        super().__init__()
        self.interpolator = Interpolator()

        self.estimation_weights = nn.Parameter(torch.ones(num_pilots), requires_grad=True)
    
    def estimate(self, Y, Xp, pilot_pos, Nfft):
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.complex64)
        if not isinstance(Xp, torch.Tensor):
            Xp = torch.tensor(Xp, dtype=torch.complex64)
        if not isinstance(pilot_pos, torch.Tensor):
            pilot_pos = torch.tensor(pilot_pos, dtype=torch.long)
        
        LS_est = torch.zeros(len(pilot_pos), dtype=torch.complex64, device=Y.device)
        for k in range(len(pilot_pos)):
            LS_est[k] = Y[pilot_pos[k]] / Xp[k]
        
        # trainable weights to LS estimates
        weighted_LS = LS_est * self.estimation_weights[:len(pilot_pos)]
        
        pilot_pos_1based = pilot_pos.float() + 1
        H_est = self.interpolator.interpolate(weighted_LS, pilot_pos_1based, Nfft)
        
        return H_est

class Interpolator(nn.Module):
    def __init__(self, method='linear', trainable=True):
        super().__init__()
        self.method = method
        self.trainable = trainable
        
        # Trainable interpolation parameters
        if self.trainable:
            self.interp_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            self.interp_beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            self.interp_gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    
    def interpolate(self, LS_est, pilot_pos_1based, Nfft):
        if self.trainable:
            # Use trainable interpolation
            H_est = torch_interpolate_trainable(
                LS_est, pilot_pos_1based, Nfft, 
                self.interp_alpha, self.interp_beta, self.interp_gamma,
                self.method
            )
        else:
            # Use non-trainable interpolation
            H_est = torch_interpolate(LS_est, pilot_pos_1based, Nfft, self.method)
        return H_est

class Equalizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Trainable scaling factor for equalization
        self.eq_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    
    def equalize(self, Y, H_est):
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.complex64)
        if not isinstance(H_est, torch.Tensor):
            H_est = torch.tensor(H_est, dtype=torch.complex64)
            
        # equalization with trainable scaling
        Y_eq = Y / (H_est * self.eq_scale)
        return Y_eq

class DataProcessor(nn.Module): # demodulation 
    def __init__(self):
        super().__init__()
    
    def process(self, Y_eq, pilot_pos, Nfft, M):
        if not isinstance(Y_eq, torch.Tensor):
            Y_eq = torch.tensor(Y_eq, dtype=torch.complex64)
        if not isinstance(pilot_pos, torch.Tensor):
            pilot_pos = torch.tensor(pilot_pos, dtype=torch.long)
        
        # Create mask for data positions
        data_mask = torch.ones(Nfft, dtype=torch.bool, device=Y_eq.device)
        data_mask[pilot_pos] = False
        Data_extracted = Y_eq[data_mask]

        rx_bits = torch_qamdemod(Data_extracted, M, mapping='Gray', outputtype='bit', unitaveragepow=True)
        
        return rx_bits

class BERCalculator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def calculate(self, rx_bits, tx_bits):
        if not isinstance(rx_bits, torch.Tensor):
            rx_bits = torch.tensor(rx_bits, dtype=torch.int32)
        if not isinstance(tx_bits, torch.Tensor):
            tx_bits = torch.tensor(tx_bits, dtype=torch.int32)
            
        bit_errors = torch.sum(rx_bits != tx_bits).float()
        BER = bit_errors / len(tx_bits)
        return BER

class BCELossCalculator(nn.Module):
    def __init__(self):
        super().__init__()

    def calculate(self, predicted_probs, true_bits):
        if not isinstance(predicted_probs, torch.Tensor):
            predicted_probs = torch.tensor(predicted_probs, dtype=torch.float32)
        if not isinstance(true_bits, torch.Tensor):
            true_bits = torch.tensor(true_bits, dtype=torch.float32)
            
        eps = 1e-12
        probs = torch.clamp(predicted_probs, eps, 1 - eps)
        bce = -torch.mean(true_bits * torch.log(probs) + (1 - true_bits) * torch.log(1 - probs))
        return bce

class ReceiverChain(nn.Module):
    
    def __init__(self, num_pilots=8):
        super().__init__()
        self.channel_estimator = ChannelEstimator(num_pilots)
        self.equalizer = Equalizer()
        self.data_processor = DataProcessor()
        self.ber_calculator = BERCalculator()
        self.bce_calculator = BCELossCalculator()
    
    def process(self, Y, Xp, pilot_pos, Nfft, M, tx_bits):
        # Channel estimation
        H_est = self.channel_estimator.estimate(Y, Xp, pilot_pos, Nfft)
        
        # Equalization
        Y_eq = self.equalizer.equalize(Y, H_est)
        
        # Data processing
        rx_bits = self.data_processor.process(Y_eq, pilot_pos, Nfft, M)
        
        # BER calculation
        BER = self.ber_calculator.calculate(rx_bits, tx_bits)
        return rx_bits, BER, H_est
    
    def calculate_bce_loss(self, predicted_probs, true_bits):
        return self.bce_calculator.calculate(predicted_probs, true_bits)

def rx_chain(Y, Xp, pilot_pos, Nfft, M, tx_bits):
    num_pilots = len(pilot_pos) if hasattr(pilot_pos, '__len__') else 8
    receiver = ReceiverChain(num_pilots)
    return receiver.process(Y, Xp, pilot_pos, Nfft, M, tx_bits) 