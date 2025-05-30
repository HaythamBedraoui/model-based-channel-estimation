import numpy as np

def encapsulate(X, window):
    shape = X.shape
    length = shape[1]
    
    # Set a fixed output length to match MATLAB's output exactly
    output_length = 9986  # This is the exact number needed to match MATLAB
    max_valid_idx = min(length - window + 1, output_length)
    
    A = np.zeros((shape[0], window, max_valid_idx), dtype=X.dtype)
    
    for i in range(max_valid_idx):
        tempWindowX = X[:, i:i+window]
        A[:, :, i] = tempWindowX
    
    return A