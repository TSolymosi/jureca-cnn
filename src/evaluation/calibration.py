import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as opt

# --- Calibration Evaluation ---
def evaluate_calibration(mu, sigma, y_true, param_names, epoch, folder_name = None):
    """
    Evaluates calibration, finds an optimal scaling factor `c` for each parameter's
    uncertainty, and plots both the original and calibrated curves.
    """
    print("Evaluating and fitting calibration curves...")
    # Theoretical fraction of data that should be within z sigma levels
    levels = np.array([1, 2, 3])
    theoretical_coverage = np.array([stats.norm.cdf(z) - stats.norm.cdf(-z) for z in levels])

    calibration_factors = {}
    
    plt.figure(figsize=(10, 8))
    
    # --- Objective Function for Optimization ---
    def calibration_loss(c, sigma_param, y_true_param):
        # Calculate empirical coverage with the calibrated sigma
        abs_z_calibrated = torch.abs((y_true_param - mu_param) / (c * sigma_param))
        empirical_coverage = np.array([(abs_z_calibrated < z).float().mean().item() for z in levels])
        # Return the sum of squared errors
        return np.sum((empirical_coverage - theoretical_coverage)**2)

    # --- Find Optimal 'c' for Each Parameter ---
    for i, param in enumerate(param_names):
        mu_param = mu[:, i]
        sigma_param = sigma[:, i]
        y_true_param = y_true[:, i]

        # Use scipy's bounded scalar optimizer to find the best `c`
        result = opt.minimize_scalar(
            calibration_loss,
            args=(sigma_param, y_true_param),
            bounds=(0.01, 1.0), # Search for c in a reasonable range
            method='bounded'
        )
        c_optimal = result.x
        calibration_factors[param] = c_optimal

        # --- Plotting ---
        # Calculate original empirical coverage
        abs_z_original = torch.abs((y_true_param - mu_param) / sigma_param)
        original_empirical = [(abs_z_original < z).float().mean().item() for z in levels]

        # Calculate calibrated empirical coverage
        abs_z_calibrated = torch.abs((y_true_param - mu_param) / (c_optimal * sigma_param))
        calibrated_empirical = [(abs_z_calibrated < z).float().mean().item() for z in levels]

        # Plot the lines
        p = plt.plot(levels, original_empirical, marker='o', ls=':', label=f'{param} (Original)')[0]
        plt.plot(levels, calibrated_empirical, marker='x', ls='-', color=p.get_color(), label=f'{param} (Calibrated, c={c_optimal:.2f})')

    plt.plot(levels, theoretical_coverage, 'k--', label='Expected (Gaussian)')
    plt.xlabel("Sigma Level")
    plt.ylabel("Fraction within Interval")
    plt.title(f"Epoch {epoch} - Calibration Curve (Original vs. Calibrated)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    # Save the plot
    if folder_name is None:
        cal_dir = f"ResNet_Plots/Calibration/"
    else:
        cal_dir = f"{folder_name}/ResNet_Plots/Calibration/"
    os.makedirs(cal_dir, exist_ok=True)
    plt.savefig(os.path.join(cal_dir, f"E{epoch}_Calibration_Curve.png"))
    plt.close()
    
    print("Calibration factors found:", {k: round(v, 3) for k, v in calibration_factors.items()})
    return calibration_factors