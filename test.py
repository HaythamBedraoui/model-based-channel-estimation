import torch
import torch.nn as nn
import numpy as np
from rx_chain_torch import ReceiverChain

def test_parameters():
    print("=== Parameter Training Test ===\n")
    receiver = ReceiverChain(num_pilots=4)
    
    # initial parameters
    print("Initial parameters:")
    initial_params = {}
    for name, param in receiver.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
            print(f"  {name}: {param.data.numpy()}")
    
    # Generate test data
    Y = torch.randn(64, dtype=torch.complex64)
    Xp = torch.randn(4, dtype=torch.complex64)
    pilot_pos = torch.tensor([8, 24, 40, 56], dtype=torch.long)
    
    # Training loop
    optimizer = torch.optim.Adam(receiver.parameters(), lr=0.1)
    
    print(f"\nTraining for 10 steps...")
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        H_est = receiver.channel_estimator.estimate(Y, Xp, pilot_pos, 64)
        Y_eq = receiver.equalizer.equalize(Y, H_est)
        loss = torch.mean(torch.abs(Y_eq)**2)
        
        loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")
    
    # final parameters
    print(f"\nFinal parameters:")
    for name, param in receiver.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.data.numpy()}")
    

if __name__ == "__main__":
    test_parameters() 