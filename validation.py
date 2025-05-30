import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

matlab_dir = "matlab_output"
python_dir = "python_output"
    
print("Validation Script")
print("-----------------")
    
# Load data
print("Loading files...")

# Load MATLAB data
matlab_perfect = sio.loadmat(os.path.join(matlab_dir, 'h_perfect_M1.mat'))['hEnc']
matlab_ls = sio.loadmat(os.path.join(matlab_dir, 'h_ls_estimation_M1.mat'))['hEncNoise']
        
# Load Python data
python_perfect = np.load(os.path.join(python_dir, 'h_perfect_M1.npy'))
python_ls = np.load(os.path.join(python_dir, 'h_ls_estimation_M1.npy'))

# Create output directory for comparison results
comparison_dir = "comparison_results"
os.makedirs(comparison_dir, exist_ok=True)

# Shape comparison
print("\n1. Shape Comparison:")
shape_match_perfect = matlab_perfect.shape == python_perfect.shape
shape_match_ls = matlab_ls.shape == python_ls.shape

shape_data = [
    ["Perfect Channel", str(matlab_perfect.shape), str(python_perfect.shape), "✓" if shape_match_perfect else "✗"],
    ["LS Estimation", str(matlab_ls.shape), str(python_ls.shape), "✓" if shape_match_ls else "✗"]
]
print(tabulate(shape_data, headers=["Dataset", "MATLAB Shape", "Python Shape", "Match"]))

if not (shape_match_perfect and shape_match_ls):
    print("ERROR: Shape mismatch detected! Cannot proceed with further comparisons.")
    exit(1)

# Statistical comparison
print("\n2. Statistical Comparison:")

# Perfect channel stats
m_perfect_mean = np.mean(np.abs(matlab_perfect))
p_perfect_mean = np.mean(np.abs(python_perfect))
m_perfect_std = np.std(np.abs(matlab_perfect))
p_perfect_std = np.std(np.abs(python_perfect))
perfect_mean_diff = abs(m_perfect_mean - p_perfect_mean)
perfect_std_diff = abs(m_perfect_std - p_perfect_std)
    
# LS estimation stats
m_ls_mean = np.mean(np.abs(matlab_ls))
p_ls_mean = np.mean(np.abs(python_ls))
m_ls_std = np.std(np.abs(matlab_ls))
p_ls_std = np.std(np.abs(python_ls))
ls_mean_diff = abs(m_ls_mean - p_ls_mean)
ls_std_diff = abs(m_ls_std - p_ls_std)

# Min/Max values
m_perfect_min, m_perfect_max = np.min(np.abs(matlab_perfect)), np.max(np.abs(matlab_perfect))
p_perfect_min, p_perfect_max = np.min(np.abs(python_perfect)), np.max(np.abs(python_perfect))
m_ls_min, m_ls_max = np.min(np.abs(matlab_ls)), np.max(np.abs(matlab_ls))
p_ls_min, p_ls_max = np.min(np.abs(python_ls)), np.max(np.abs(python_ls))

stats_data = [
    ["Perfect Channel Mean", f"{m_perfect_mean:.6f}", f"{p_perfect_mean:.6f}", f"{perfect_mean_diff:.6f}"],
    ["Perfect Channel Std", f"{m_perfect_std:.6f}", f"{p_perfect_std:.6f}", f"{perfect_std_diff:.6f}"],
    ["Perfect Channel Min", f"{m_perfect_min:.6f}", f"{p_perfect_min:.6f}", f"{abs(m_perfect_min - p_perfect_min):.6f}"],
    ["Perfect Channel Max", f"{m_perfect_max:.6f}", f"{p_perfect_max:.6f}", f"{abs(m_perfect_max - p_perfect_max):.6f}"],
    ["LS Estimation Mean", f"{m_ls_mean:.6f}", f"{p_ls_mean:.6f}", f"{ls_mean_diff:.6f}"],
    ["LS Estimation Std", f"{m_ls_std:.6f}", f"{p_ls_std:.6f}", f"{ls_std_diff:.6f}"],
    ["LS Estimation Min", f"{m_ls_min:.6f}", f"{p_ls_min:.6f}", f"{abs(m_ls_min - p_ls_min):.6f}"],
    ["LS Estimation Max", f"{m_ls_max:.6f}", f"{p_ls_max:.6f}", f"{abs(m_ls_max - p_ls_max):.6f}"]
]
print(tabulate(stats_data, headers=["Metric", "MATLAB", "Python", "Absolute Difference"]))

# Correlation and error metrics
print("\n3. Correlation and Error Metrics:")

perfect_corr = np.corrcoef(np.abs(matlab_perfect.flatten()), np.abs(python_perfect.flatten()))[0, 1]
ls_corr = np.corrcoef(np.abs(matlab_ls.flatten()), np.abs(python_ls.flatten()))[0, 1]

perfect_mae = np.mean(np.abs(matlab_perfect - python_perfect))
ls_mae = np.mean(np.abs(matlab_ls - python_ls))

perfect_mse = np.mean(np.abs(matlab_perfect - python_perfect)**2)
ls_mse = np.mean(np.abs(matlab_ls - python_ls)**2)

# Normalized root mean square error (NRMSE)
perfect_nrmse = np.sqrt(perfect_mse) / (np.max(np.abs(matlab_perfect)) - np.min(np.abs(matlab_perfect)))
ls_nrmse = np.sqrt(ls_mse) / (np.max(np.abs(matlab_ls)) - np.min(np.abs(matlab_ls)))

metrics_data = [
    ["Correlation", f"{perfect_corr:.6f}", f"{ls_corr:.6f}"],
    ["Mean Absolute Error (MAE)", f"{perfect_mae:.6f}", f"{ls_mae:.6f}"],
    ["Mean Squared Error (MSE)", f"{perfect_mse:.6f}", f"{ls_mse:.6f}"],
    ["Normalized RMSE", f"{perfect_nrmse:.6f}", f"{ls_nrmse:.6f}"]
]
print(tabulate(metrics_data, headers=["Metric", "Perfect Channel", "LS Estimation"]))

# Element-wise comparison (sample)
print("\n4. Element-wise Sample Comparison (first 5 elements):")
sample_size = min(5, matlab_perfect.size)
sample_indices = np.random.choice(matlab_perfect.size, sample_size, replace=False)

sample_data = []
for idx in sample_indices:
    flat_idx = np.unravel_index(idx, matlab_perfect.shape)
    m_perfect_val = matlab_perfect[flat_idx]
    p_perfect_val = python_perfect[flat_idx]
    m_ls_val = matlab_ls[flat_idx]
    p_ls_val = python_ls[flat_idx]
    
    sample_data.append([
        str(flat_idx),
        f"{m_perfect_val:.6f}", 
        f"{p_perfect_val:.6f}",
        f"{abs(m_perfect_val - p_perfect_val):.6f}",
        f"{m_ls_val:.6f}",
        f"{p_ls_val:.6f}",
        f"{abs(m_ls_val - p_ls_val):.6f}"
    ])

print(tabulate(sample_data, headers=["Index", "MATLAB Perfect", "Python Perfect", "Diff", "MATLAB LS", "Python LS", "Diff"]))

# Visualizations
print("\n5. Generating Visualizations...")

# Histogram comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(np.abs(matlab_perfect.flatten()), bins=50, alpha=0.7, label='MATLAB')
plt.hist(np.abs(python_perfect.flatten()), bins=50, alpha=0.7, label='Python')
plt.title('Perfect Channel Magnitude Distribution')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(np.abs(matlab_ls.flatten()), bins=50, alpha=0.7, label='MATLAB')
plt.hist(np.abs(python_ls.flatten()), bins=50, alpha=0.7, label='Python')
plt.title('LS Estimation Magnitude Distribution')
plt.legend()

# Scatter plots
plt.subplot(2, 2, 3)
plt.scatter(np.abs(matlab_perfect.flatten()), np.abs(python_perfect.flatten()), alpha=0.1)
plt.xlabel('MATLAB Perfect Channel')
plt.ylabel('Python Perfect Channel')
plt.title(f'Correlation: {perfect_corr:.4f}')
plt.plot([0, np.max(np.abs(matlab_perfect))], [0, np.max(np.abs(matlab_perfect))], 'r--')

plt.subplot(2, 2, 4)
plt.scatter(np.abs(matlab_ls.flatten()), np.abs(python_ls.flatten()), alpha=0.1)
plt.xlabel('MATLAB LS Estimation')
plt.ylabel('Python LS Estimation')
plt.title(f'Correlation: {ls_corr:.4f}')
plt.plot([0, np.max(np.abs(matlab_ls))], [0, np.max(np.abs(matlab_ls))], 'r--')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'distribution_comparison.png'))

# Error distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
error_perfect = np.abs(matlab_perfect - python_perfect).flatten()
plt.hist(error_perfect, bins=50)
plt.title(f'Perfect Channel Error Distribution\nMAE: {perfect_mae:.6f}')
plt.xlabel('Absolute Error')

plt.subplot(1, 2, 2)
error_ls = np.abs(matlab_ls - python_ls).flatten()
plt.hist(error_ls, bins=50)
plt.title(f'LS Estimation Error Distribution\nMAE: {ls_mae:.6f}')
plt.xlabel('Absolute Error')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'error_distribution.png'))

# Heatmap of differences (for a 2D slice if data is higher dimensional)
if len(matlab_perfect.shape) > 2:
    # Take a 2D slice for visualization
    m_perfect_slice = np.abs(matlab_perfect[0, :, :])
    p_perfect_slice = np.abs(python_perfect[0, :, :])
    m_ls_slice = np.abs(matlab_ls[0, :, :])
    p_ls_slice = np.abs(python_ls[0, :, :])
else:
    # Use the full data if it's already 2D
    m_perfect_slice = np.abs(matlab_perfect)
    p_perfect_slice = np.abs(python_perfect)
    m_ls_slice = np.abs(matlab_ls)
    p_ls_slice = np.abs(python_ls)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(m_perfect_slice, aspect='auto')
plt.colorbar()
plt.title('MATLAB Perfect Channel (Magnitude)')

plt.subplot(2, 2, 2)
plt.imshow(p_perfect_slice, aspect='auto')
plt.colorbar()
plt.title('Python Perfect Channel (Magnitude)')

plt.subplot(2, 2, 3)
plt.imshow(m_ls_slice, aspect='auto')
plt.colorbar()
plt.title('MATLAB LS Estimation (Magnitude)')

plt.subplot(2, 2, 4)
plt.imshow(p_ls_slice, aspect='auto')
plt.colorbar()
plt.title('Python LS Estimation (Magnitude)')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'magnitude_comparison.png'))

# Difference heatmaps
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(m_perfect_slice - p_perfect_slice), aspect='auto')
plt.colorbar()
plt.title('Perfect Channel Absolute Difference')

plt.subplot(1, 2, 2)
plt.imshow(np.abs(m_ls_slice - p_ls_slice), aspect='auto')
plt.colorbar()
plt.title('LS Estimation Absolute Difference')

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'difference_heatmap.png'))

# Summary
print("\n6. Summary:")
threshold = 1e-5  # Threshold for considering values as matching

perfect_match_percent = 100 * np.mean(np.abs(matlab_perfect - python_perfect) < threshold)
ls_match_percent = 100 * np.mean(np.abs(matlab_ls - python_ls) < threshold)

summary_data = [
    ["Perfect Channel", f"{perfect_corr:.6f}", f"{perfect_mae:.6f}", f"{perfect_match_percent:.2f}%"],
    ["LS Estimation", f"{ls_corr:.6f}", f"{ls_mae:.6f}", f"{ls_match_percent:.2f}%"]
]
print(tabulate(summary_data, headers=["Dataset", "Correlation", "MAE", "% Elements Matching"]))

print(f"\nDetailed comparison results saved to '{comparison_dir}' directory.")
print("\nValidation complete")
