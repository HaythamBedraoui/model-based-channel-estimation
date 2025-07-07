import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from rx_chain_torch import ReceiverChain

class TrainingLoop:
    def __init__(self, num_pilots=8, learning_rate=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.receiver = ReceiverChain(num_pilots).to(self.device)
        self.optimizer = optim.Adam(self.receiver.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_losses = []
        self.train_bers = []
        
    def trainingSample(self, snr_db=20, Nfft=64, Nps=4, M=4):
        Ndata = Nfft - Nps
        tx_bits = np.random.randint(0, 2, Ndata * int(np.log2(M)))
        
        pilot_symbols = (2*np.random.randint(0, 2, Nps) - 1) + 1j*(2*np.random.randint(0, 2, Nps) - 1)
        pilot_symbols = pilot_symbols / np.sqrt(2)
        
        data_real = 2*tx_bits[0::2] - 1
        data_imag = 2*tx_bits[1::2] - 1
        data_symbols = (data_real + 1j*data_imag) / np.sqrt(2)
        
        pilot_pos = np.linspace(0, Nfft-1, Nps, dtype=int)
        
        X = np.zeros(Nfft, dtype=complex)
        X[pilot_pos] = pilot_symbols
        
        data_mask = np.ones(Nfft, dtype=bool)
        data_mask[pilot_pos] = False
        X[data_mask] = data_symbols
        
        H = (np.random.randn(Nfft) + 1j*np.random.randn(Nfft)) / np.sqrt(2)
        
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(Nfft) + 1j*np.random.randn(Nfft))
        Y = H * X + noise
        
        return {
            'Y': Y,
            'Xp': pilot_symbols,
            'pilot_pos': pilot_pos,
            'Nfft': Nfft,
            'M': M,
            'tx_bits': tx_bits
        }
    
    def convert_bits_to_probabilities(self, rx_bits):
        return torch.sigmoid(rx_bits.float())
    
    def train_one_epoch(self, num_samples=100, snr_range=(10, 25)):
        epoch_loss = 0.0
        epoch_ber = 0.0
        self.receiver.train()
        
        for sample_idx in range(num_samples):
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            sample = self.generate_training_sample(snr_db=snr_db)
            
            Y = torch.tensor(sample['Y'], dtype=torch.complex64).to(self.device)
            Xp = torch.tensor(sample['Xp'], dtype=torch.complex64).to(self.device)
            pilot_pos = torch.tensor(sample['pilot_pos'], dtype=torch.long).to(self.device)
            tx_bits = torch.tensor(sample['tx_bits'], dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            
            rx_bits, ber, H_est = self.receiver.process(
                Y, Xp, pilot_pos, sample['Nfft'], sample['M'], sample['tx_bits']
            )
            
            rx_probs = self.convert_bits_to_probabilities(rx_bits)
            loss = self.criterion(rx_probs, tx_bits)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ber += ber.item()
        
        return epoch_loss / num_samples, epoch_ber / num_samples
    
    def train(self, num_epochs=50, samples_per_epoch=100):
        print("Starting training...")
        print("Trainable parameters:")
        for name, param in self.receiver.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
        
        for epoch in range(num_epochs):
            if epoch < 15:
                snr_range = (20, 30)
            elif epoch < 35:
                snr_range = (10, 25)
            else:
                snr_range = (5, 15)
            
            avg_loss, avg_ber = self.train_one_epoch(num_samples=samples_per_epoch, snr_range=snr_range)
            
            self.train_losses.append(avg_loss)
            self.train_bers.append(avg_ber)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, BER={avg_ber:.4f}, SNR={snr_range}")
                for name, param in self.receiver.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}: {param.data.cpu().numpy()}")
        
        return self.train_losses, self.train_bers
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss')
        ax1.grid(True)
        
        ax2.semilogy(self.train_bers)
        ax2.set_title('Training BER')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BER')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    trainer = TrainingLoop(num_pilots=8, learning_rate=0.01)
    train_losses, train_bers = trainer.train(num_epochs=50, samples_per_epoch=50)
    trainer.plot_training_history()
    torch.save(trainer.receiver.state_dict(), 'trained_receiver.pth')
    print("Training completed. Model saved as 'trained_receiver.pth'")

if __name__ == "__main__":
    main() 