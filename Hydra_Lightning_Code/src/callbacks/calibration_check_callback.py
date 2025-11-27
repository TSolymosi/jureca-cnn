import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class CalibrationCheckCallback(Callback):
    """
    Checks if predicted uncertainties are well-calibrated.
    
    CRITICAL FIX: Z-scores are ALWAYS calculated in scaled space where the 
    Gaussian assumption holds. Physical space is only for visualization.
    """
    def __init__(self, 
                 model_params,
                 log_scale_params,
                 output_dir="./calibration_check",
                 check_every_n_epochs=5,
                 snr_bins=[(0, 15), (15, 25), (25, 50), (50, np.inf)]):
        super().__init__()
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.output_dir = output_dir
        self.check_every_n_epochs = check_every_n_epochs
        self.snr_bins = snr_bins


    
    
    
    
    # ... rest of your calibration callback ...
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check calibration at end of validation."""
        epoch = trainer.current_epoch
        if (epoch + 1) % self.check_every_n_epochs != 0:
            return
        
        if trainer.sanity_checking:
            return
        
        
        # ===== COMPREHENSIVE DEBUGGING =====
        print(f"\n{'='*100}")
        print(f"VALIDATION SAMPLE ACCOUNTING - Epoch {epoch+1}")
        print(f"{'='*100}")
        
        # 1. Check the dataloader size
        val_loader = trainer.val_dataloaders
        if val_loader:
            dataset_size = len(val_loader.dataset)
            batch_size = val_loader.batch_size
            expected_batches = (dataset_size + batch_size - 1) // batch_size
            print(f"Validation dataset size: {dataset_size}")
            print(f"Batch size: {batch_size}")
            print(f"Expected batches: {expected_batches}")
        
        # 2. Check the lightning module's outputs
        if hasattr(pl_module, 'validation_step_outputs'):
            num_batches = len(pl_module.validation_step_outputs)
            print(f"Actual batches in validation_step_outputs: {num_batches}")
            
            if num_batches > 0:
                total_in_outputs = sum(
                    o.get('targets', torch.empty(0)).shape[0] 
                    for o in pl_module.validation_step_outputs 
                    if isinstance(o, dict) and 'targets' in o
                )
                print(f"Total samples in validation_step_outputs: {total_in_outputs}")
        
        # 3. Check SNR exclusions
        if hasattr(pl_module, '_val_excluded_count'):
            print(f"Samples excluded by SNR filter: {pl_module._val_excluded_count}")
        
        # 4. Check the cache
        cache = getattr(pl_module, "_val_cache", {})
        if cache:
            for key in ['preds', 'targets', 'sigmas', 'snr']:
                if key in cache and cache[key] is not None:
                    print(f"Cache['{key}'] shape: {cache[key].shape}")
        else:
            print("Cache is empty!")
        
        print(f"{'='*100}\n")
        
        
        ds = getattr(pl_module, "_val_dataset", None)
        
        preds_scaled = cache.get("preds", None)
        targets_scaled = cache.get("targets", None)
        sigmas_scaled = cache.get("sigmas", None)
        snr = cache.get("snr", None)
        
        if preds_scaled is None or targets_scaled is None or sigmas_scaled is None or ds is None:
            print("[CalibrationCheck] Missing data, skipping.")
            return
        
        # Debug: Check for collapsed uncertainties
        print("\n[DEBUG] Scaled space statistics:")
        for i, param in enumerate(self.model_params):
            sigma_mean = sigmas_scaled[:, i].mean().item()
            sigma_min = sigmas_scaled[:, i].min().item()
            sigma_max = sigmas_scaled[:, i].max().item()
            
            # Count collapsed samples
            n_collapsed = (sigmas_scaled[:, i] < 0.00011).sum().item()
            
            print(f"{param}:")
            print(f"  σ_scaled: mean={sigma_mean:.6f}, min={sigma_min:.6f}, max={sigma_max:.6f}")
            print(f"  Collapsed samples (σ<0.00011): {n_collapsed}/{len(sigmas_scaled)}")
        
        # CRITICAL: Floor the sigmas to prevent division by near-zero
        sigmas_scaled_safe = torch.clamp(sigmas_scaled, min=1e-3)  # Floor at 0.001
        
        # Calculate z-scores with safe sigmas (ALWAYS in scaled space!)
        z_scores_scaled = (preds_scaled - targets_scaled) / sigmas_scaled_safe
        
        print("\n[DEBUG] Z-scores after sigma floor:")
        for i, param in enumerate(self.model_params):
            z_mean = z_scores_scaled[:, i].mean().item()
            z_std = z_scores_scaled[:, i].std().item()
            print(f"{param}: z_mean={z_mean:.3f}, z_std={z_std:.3f}")
        
        # Prepare data for visualization in different spaces
        # ================================================================
        # SCALED SPACE (z-score normalized): For calculations
        # ================================================================
        mu_scaled = preds_scaled
        y_scaled = targets_scaled
        sigma_scaled = sigmas_scaled_safe
        
        # ================================================================
        # LOG SPACE (for log-params) / PHYSICAL SPACE (for linear params)
        # ================================================================
        # Simply undo the z-score normalization
        mu_logspace = preds_scaled * ds.scaler_stds + ds.scaler_means
        y_logspace = targets_scaled * ds.scaler_stds + ds.scaler_means
        sigma_logspace = sigmas_scaled_safe * ds.scaler_stds
        
        # ================================================================
        # PHYSICAL SPACE: Exponentiate log-params for interpretability
        # (Only for visualization! Don't use these sigmas for z-scores!)
        # ================================================================
        targets_orig = ds.inverse_transform_labels(targets_scaled)
        preds_orig, sigmas_orig = ds.inverse_transform_labels_with_uncertainty(
            preds_scaled, sigmas_scaled_safe
        )
        
        # Call analyze_calibration with ALL required arguments
        self._analyze_calibration(
            mu_orig=preds_orig,           # Predictions in physical space
            sigma_orig=sigmas_orig,        # Uncertainties in physical space
            y_orig=targets_orig,           # Targets in physical space
            mu_scaled=mu_scaled,           # Predictions in scaled space
            y_scaled=y_scaled,             # Targets in scaled space
            sigma_scaled=sigma_scaled,     # Uncertainties in scaled space
            z_scores_scaled=z_scores_scaled,  # Z-scores (calculated in scaled space)
            snr=snr,                       # SNR values
            epoch=epoch                    # Current epoch
        )
    
    def _analyze_calibration(self, mu_orig, sigma_orig, y_orig,
                            mu_scaled, y_scaled, sigma_scaled,
                            z_scores_scaled, snr, epoch):
        """Analyze and visualize calibration."""
        
        # Expected coverage rates
        sigma_levels = np.array([1, 2, 3])
        expected_coverage = np.array([0.6827, 0.9545, 0.9973])
        
        # Calculate actual coverage for each parameter
        results = {}
        for i, param_name in enumerate(self.model_params):
            z_i = z_scores_scaled[:, i].abs()
            actual_coverage = np.array([
                (z_i <= level).float().mean().item() 
                for level in sigma_levels
            ])
            
            # SNR-binned metrics
            snr_binned_metrics = self._calculate_snr_binned_metrics(
                z_scores_scaled[:, i], snr, sigma_levels
            ) if snr is not None else None

            # --- START: MODIFIED RESIDUAL CALCULATION ---
            # Calculate the correct type of residual based on parameter scaling
            y_i_orig = y_orig[:, i]
            mu_i_orig = mu_orig[:, i]
            
            if param_name in self.log_scale_params:
                # For log-params, use RELATIVE residual to be scale-invariant
                mask = torch.abs(y_i_orig) > 1e-9
                residuals_for_plotting = torch.full_like(y_i_orig, float('nan'))
                residuals_for_plotting[mask] = (mu_i_orig[mask] - y_i_orig[mask]) / y_i_orig[mask]
            else:
                # For linear-params, ABSOLUTE residual is fine
                residuals_for_plotting = mu_i_orig - y_i_orig
            # --- END: MODIFIED RESIDUAL CALCULATION ---

            
            results[param_name] = {
                'actual': actual_coverage,
                'expected': expected_coverage,
                'z_scores': z_scores_scaled[:, i].cpu().numpy(),
                # Store both scaled and original for different visualizations
                'sigma_scaled': sigma_scaled[:, i].cpu().numpy(),
                'sigma_orig': sigma_orig[:, i].cpu().numpy(),
                'residuals_orig': residuals_for_plotting.cpu().numpy(), # Use the corrected residual
                'residuals_scaled': (mu_scaled[:, i] - y_scaled[:, i]).cpu().numpy(),
                'true_values_orig': y_orig[:, i].cpu().numpy(),
                'pred_values_orig': mu_orig[:, i].cpu().numpy(),
                'true_values_scaled': y_scaled[:, i].cpu().numpy(),
                'pred_values_scaled': mu_scaled[:, i].cpu().numpy(),
                'snr_binned': snr_binned_metrics
            }
        
        # Create visualizations
        self._plot_calibration_curves(results, epoch)
        self._plot_z_score_histograms(results, epoch)
        self._plot_per_parameter_summary(results, epoch)
        self._plot_residuals_vs_true_symlog(results, epoch) 
        
        # SNR-aware plots
        if snr is not None:
            self._plot_snr_calibration_curves(results, epoch, snr)
            self._plot_sigma_vs_value_by_snr(results, epoch, snr.cpu().numpy())
            self._plot_residuals_vs_sigma_by_snr(results, epoch, snr.cpu().numpy())
        
        # Print summary
        self._print_calibration_summary(results, epoch)
    
    def _calculate_snr_binned_metrics(self, z_scores_param, snr, sigma_levels):
        """Calculate calibration metrics within each SNR bin."""
        snr_np = snr.cpu().numpy() if torch.is_tensor(snr) else snr
        z_np = z_scores_param.cpu().numpy() if torch.is_tensor(z_scores_param) else z_scores_param
        
        binned_results = {}
        
        for snr_min, snr_max in self.snr_bins:
            mask = (snr_np >= snr_min) & (snr_np < snr_max)
            n_samples = mask.sum()
            
            if n_samples < 10:
                continue
            
            z_binned = z_np[mask]
            coverage = np.array([
                (np.abs(z_binned) <= level).mean() 
                for level in sigma_levels
            ])
            
            bin_name = f"{snr_min}-{snr_max if snr_max != np.inf else 'inf'}"
            binned_results[bin_name] = {
                'coverage': coverage,
                'z_mean': z_binned.mean(),
                'z_std': z_binned.std(),
                'n_samples': n_samples
            }
        
        return binned_results
    
    def _plot_calibration_curves(self, results, epoch):
        """Plot coverage vs sigma level for all parameters."""
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        sigma_levels = np.array([1, 2, 3])
        expected = results[self.model_params[0]]['expected']
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            actual = results[param_name]['actual']
            
            ax.plot(sigma_levels, expected * 100, 'k--', linewidth=2, 
                   label='Expected (ideal)', alpha=0.7)
            ax.plot(sigma_levels, actual * 100, 'o-', linewidth=2, 
                   label='Actual', markersize=8)
            
            ax.axhline(68.27, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(95.45, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(99.73, color='gray', linestyle=':', alpha=0.5)
            
            coverage_1sig = actual[0] * 100
            deviation = coverage_1sig - 68.27
            color = 'green' if abs(deviation) < 5 else 'orange' if abs(deviation) < 10 else 'red'
            ax.text(0.05, 0.95, f'1σ: {coverage_1sig:.1f}%\n(Δ{deviation:+.1f}%)',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                   fontsize=9)
            
            ax.set_xlabel('Sigma Level', fontsize=10)
            ax.set_ylabel('Coverage (%)', fontsize=10)
            ax.set_title(param_name, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Calibration Curves', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_calibration_curves.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_z_score_histograms(self, results, epoch):
        """Plot z-score distributions."""
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            z_scores = results[param_name]['z_scores']
            
            # Filter to reasonable range for visualization
            z_plot = z_scores[(z_scores > -10) & (z_scores < 10)]
            
            # Calculate ROBUST statistics
            z_median = np.median(z_scores)
            z_q25, z_q75 = np.percentile(z_scores, [25, 75])
            z_iqr = z_q75 - z_q25
            
            # Standard statistics for comparison
            z_mean = z_scores.mean()
            z_std = z_scores.std()
            
            # Count outliers
            n_outliers = ((z_scores < -10) | (z_scores > 10)).sum()
            pct_outliers = 100 * n_outliers / len(z_scores)
            
            ax.hist(z_plot, bins=50, density=True, alpha=0.7, 
                edgecolor='black', label='Observed')
            
            # Overlay ideal N(0,1)
            x = np.linspace(-4, 4, 100)
            ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), 
                'r-', linewidth=2, label='N(0,1) ideal')
            
            # Determine status
            if abs(z_median) < 0.1 and abs(z_iqr/1.35 - 1.0) < 0.2:  # IQR/1.35 ≈ σ for normal
                color = 'green'
            elif abs(z_median) < 0.5 and pct_outliers < 5:
                color = 'orange'
            else:
                color = 'red'
            
            ax.set_xlabel('Z-Score (in scaled space)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            
            # Updated title with robust statistics
            title = f'{param_name}\n'
            title += f'median={z_median:.2f}, IQR={z_iqr:.2f}\n'
            title += f'(μ={z_mean:.2f}, σ={z_std:.2f})\n'
            title += f'{pct_outliers:.1f}% outliers'
            
            ax.set_title(title, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-4, 4])
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Z-Score Distributions (Scaled Space)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_zscore_histograms.png'), 
                dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_per_parameter_summary(self, results, epoch):
        """Summary bar charts."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        param_names = self.model_params
        n_params = len(param_names)
        
        coverage_1sig = np.array([results[p]['actual'][0] for p in param_names])
        z_means = np.array([results[p]['z_scores'].mean() for p in param_names])
        z_stds = np.array([results[p]['z_scores'].std() for p in param_names])
        
        x = np.arange(n_params)
        
        # Coverage
        ax1 = axes[0]
        bars = ax1.bar(x, coverage_1sig * 100, alpha=0.8)
        for bar, cov in zip(bars, coverage_1sig):
            if abs(cov - 0.6827) < 0.05:
                bar.set_color('green')
            elif abs(cov - 0.6827) < 0.10:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        ax1.axhline(68.27, color='black', linestyle='--', linewidth=2)
        ax1.set_ylabel('Coverage (%)')
        ax1.set_title('1-Sigma Coverage')
        ax1.set_xticks(x)
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 105])
        
        # Bias
        ax2 = axes[1]
        bars = ax2.bar(x, z_means, alpha=0.8)
        for bar, zm in zip(bars, z_means):
            if abs(zm) < 0.1:
                bar.set_color('green')
            elif abs(zm) < 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        ax2.axhline(0, color='black', linestyle='--', linewidth=2)
        ax2.set_ylabel('Mean Z-Score')
        ax2.set_title('Z-Score Bias\n(should be ~0)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Spread
        ax3 = axes[2]
        bars = ax3.bar(x, z_stds, alpha=0.8)
        for bar, zs in zip(bars, z_stds):
            if abs(zs - 1.0) < 0.2:
                bar.set_color('green')
            elif abs(zs - 1.0) < 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        ax3.axhline(1.0, color='black', linestyle='--', linewidth=2)
        ax3.set_ylabel('Std of Z-Scores')
        ax3.set_title('Z-Score Spread\n(should be ~1)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(param_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Epoch {epoch+1}: Calibration Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_calibration_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_snr_calibration_curves(self, results, epoch, snr):
        """SNR-conditional calibration curves."""
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        sigma_levels = np.array([1, 2, 3])
        bin_colors = plt.cm.viridis(np.linspace(0, 1, len(self.snr_bins)))
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            snr_binned = results[param_name].get('snr_binned', None)
            
            if snr_binned is None or len(snr_binned) == 0:
                ax.text(0.5, 0.5, 'No SNR data', transform=ax.transAxes, ha='center')
                ax.set_title(param_name)
                continue
            
            ax.plot(sigma_levels, [68.27, 95.45, 99.73], 'k--', linewidth=2, label='Expected')
            
            for (bin_name, metrics), color in zip(snr_binned.items(), bin_colors):
                coverage = metrics['coverage'] * 100
                n_samples = metrics['n_samples']
                ax.plot(sigma_levels, coverage, 'o-', linewidth=2, 
                       label=f'SNR {bin_name} (n={n_samples})', color=color, markersize=6)
            
            ax.set_xlabel('Sigma Level')
            ax.set_ylabel('Coverage (%)')
            ax.set_title(f'{param_name} - SNR-Binned', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: SNR-Conditional Calibration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_snr_calibration.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sigma_vs_value_by_snr(self, results, epoch, snr):
        """Plot predicted sigma vs true value, colored by SNR."""
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        vmin, vmax = 10, 50
        norm = Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))
        cmap = plt.cm.viridis
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            true_vals = results[param_name]['true_values_orig']
            sigma_vals = results[param_name]['sigma_orig']
            
            snr_log = np.log10(np.clip(snr, vmin, vmax))
            
            scatter = ax.scatter(true_vals, sigma_vals, c=snr_log, cmap=cmap,
                               alpha=0.5, s=20, norm=norm)
            
            if param_name in self.log_scale_params:
                ax.set_xscale('log')
                ax.set_yscale('log')
            
            ax.set_xlabel(f'True {param_name}')
            ax.set_ylabel(f'Predicted σ')
            ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                cbar = plt.colorbar(scatter, ax=ax, label='SNR')
                cbar_ticks = [10, 15, 20, 30, 50]
                cbar.set_ticks(np.log10(cbar_ticks))
                cbar.set_ticklabels(cbar_ticks)
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Uncertainty vs True Value (by SNR)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_sigma_vs_value_snr.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    
    
    

    def _plot_residuals_vs_sigma_by_snr_old(self, results, epoch, snr):
        """
        Plot residuals vs predicted sigma, colored by SNR.
        Uses symmetric log scale for residuals of log-parameters.
        
        FIXED: Dynamic y-axis limits and ticks based on actual data range.
        """
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        vmin, vmax = 10, 50
        norm = Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))
        cmap = plt.cm.viridis
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            residuals = results[param_name]['residuals_orig']
            sigma_vals = results[param_name]['sigma_orig']
            
            snr_log = np.log10(np.clip(snr, vmin, vmax))
            
            if param_name in self.log_scale_params:
                # === SYMMETRIC LOG SCALE for LOG PARAMETERS ===
                
                # Transform residuals: sign(r) * log10(|r| + epsilon)
                epsilon = 1e-3
                residuals_symlog = np.sign(residuals) * np.log10(np.abs(residuals) + epsilon)
                
                # Scatter plot with SNR coloring
                scatter = ax.scatter(sigma_vals, residuals_symlog, c=snr_log, cmap=cmap,
                                alpha=0.5, s=20, norm=norm)
                
                # Zero line
                ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # ±1σ reference lines in transformed space
                ref_1sigma_pos = np.log10(1.0 + epsilon)
                ref_1sigma_neg = -np.log10(1.0 + epsilon)
                ax.axhline(ref_1sigma_pos, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                ax.axhline(ref_1sigma_neg, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                
                # === DYNAMIC Y-AXIS TICKS ===
                # Get data range
                y_min, y_max = residuals_symlog.min(), residuals_symlog.max()
                y_range = y_max - y_min
                
                # Create candidate tick values (in original residual space)
                candidate_values = []
                for exp in range(-3, 3):  # 0.001 to 100
                    base = 10.0 ** exp
                    candidate_values.extend([
                        -10 * base, -3 * base, -base, -0.3 * base, -0.1 * base,
                        0.1 * base, 0.3 * base, base, 3 * base, 10 * base
                    ])
                candidate_values.append(0.0)
                candidate_values = np.array(sorted(set(candidate_values)))
                
                # Transform to symlog space
                tick_positions = np.sign(candidate_values) * np.log10(np.abs(candidate_values) + epsilon)
                
                # Filter to visible range with margin
                margin = y_range * 0.1
                visible_mask = (tick_positions >= y_min - margin) & (tick_positions <= y_max + margin)
                
                # Select ticks
                filtered_positions = tick_positions[visible_mask]
                filtered_values = candidate_values[visible_mask]
                
                # If too many ticks, subsample
                if len(filtered_positions) > 10:
                    stride = max(1, len(filtered_positions) // 8)
                    indices = np.arange(0, len(filtered_positions), stride)
                    filtered_positions = filtered_positions[indices]
                    filtered_values = filtered_values[indices]
                
                # Ensure zero is included if in range
                if y_min < 0 < y_max and 0.0 not in filtered_values:
                    zero_symlog = 0.0
                    insert_idx = np.searchsorted(filtered_positions, zero_symlog)
                    filtered_positions = np.insert(filtered_positions, insert_idx, zero_symlog)
                    filtered_values = np.insert(filtered_values, insert_idx, 0.0)
                
                # Set ticks
                ax.set_yticks(filtered_positions)
                
                # Format labels
                labels = []
                for v in filtered_values:
                    if v == 0:
                        labels.append('0')
                    elif abs(v) >= 1:
                        labels.append(f'{v:+.0f}')
                    elif abs(v) >= 0.1:
                        labels.append(f'{v:+.1f}')
                    elif abs(v) >= 0.01:
                        labels.append(f'{v:+.2f}')
                    else:
                        labels.append(f'{v:+.3f}')
                ax.set_yticklabels(labels, fontsize=8)
                
                # Set y-limits with margin
                ax.set_ylim(y_min - margin, y_max + margin)
                
                # Log scale for x-axis (sigma)
                ax.set_xscale('log')
                
                # Labels
                ax.set_xlabel(f'Predicted σ_log({param_name})', fontsize=10)
                ax.set_ylabel(f'Residual: Δlog({param_name})\n[symlog scale]', fontsize=10)
                
                # Add explanation text box
                textstr = 'Y: sign(Δ)×log₁₀(|Δ|+ε)\nΔ = log(pred)−log(true)'
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            else:
                # === STANDARD LINEAR PLOT for LINEAR PARAMETERS ===
                
                scatter = ax.scatter(sigma_vals, residuals, c=snr_log, cmap=cmap,
                                alpha=0.5, s=20, norm=norm)
                
                # Zero line
                ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # ±1σ reference (should contain ~68% of points ideally)
                residual_std = np.std(residuals)
                ax.axhline(residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                ax.axhline(-residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                
                # Labels
                ax.set_xlabel(f'Predicted σ_{param_name}', fontsize=10)
                ax.set_ylabel(f'Residual: pred - true', fontsize=10)
                
                # Let matplotlib auto-scale for linear params
            
            ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to first plot only
            if i == 0:
                cbar = plt.colorbar(scatter, ax=ax, label='SNR')
                cbar_ticks = [10, 15, 20, 30, 50]
                cbar.set_ticks(np.log10(cbar_ticks))
                cbar.set_ticklabels(cbar_ticks)
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Residuals vs Sigma (by SNR)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_residuals_vs_sigma_snr.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_calibration_summary(self, results, epoch):
        """Print summary to console."""
        print(f"\n{'='*100}")
        print(f"CALIBRATION CHECK - Epoch {epoch+1}")
        print(f"{'='*100}")
        print(f"{'Parameter':<12} {'1σ Cov':<10} {'2σ Cov':<10} {'Z-μ':<10} {'Z-σ':<10} {'Status':<15}")
        print(f"{'-'*100}")
        
        for param_name in self.model_params:
            cov_1 = results[param_name]['actual'][0]
            cov_2 = results[param_name]['actual'][1]
            z_mean = results[param_name]['z_scores'].mean()
            z_std = results[param_name]['z_scores'].std()
            
            cov_good = abs(cov_1 - 0.6827) < 0.05
            bias_good = abs(z_mean) < 0.1
            spread_good = abs(z_std - 1.0) < 0.2
            
            if cov_good and bias_good and spread_good:
                status = "✓ GOOD"
            elif abs(cov_1 - 0.6827) < 0.15 and abs(z_mean) < 0.5:
                status = "~ OK"
            else:
                status = "✗ BAD"
            
            print(f"{param_name:<12} {cov_1*100:>6.2f}%  {cov_2*100:>6.2f}%  "
                  f"{z_mean:>8.2f}  {z_std:>8.2f}  {status:<15}")
            
            snr_binned = results[param_name].get('snr_binned', None)
            if snr_binned:
                for bin_name, metrics in snr_binned.items():
                    cov_bin = metrics['coverage'][0] * 100
                    z_mean_bin = metrics['z_mean']
                    z_std_bin = metrics['z_std']
                    n = metrics['n_samples']
                    print(f"  └─ SNR {bin_name:<10} {cov_bin:>6.2f}%  {'-'*8}  "
                          f"{z_mean_bin:>8.2f}  {z_std_bin:>8.2f}  (n={n})")
        
        print(f"{'='*100}\n")

    def _plot_residuals_vs_true_symlog_old(self, results, epoch):
            """
            Plot residuals vs true values with symmetric log scaling for log-parameters.
            
            FIXED: Dynamic y-axis limits and ticks based on actual data range.
            """
            n_params = len(self.model_params)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, param_name in enumerate(self.model_params):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                
                true_vals = results[param_name]['true_values_orig']
                residuals = results[param_name]['residuals_orig']
                
                # Subsample if too many points
                n_samples = len(true_vals)
                if n_samples > 2000:
                    indices = np.random.choice(n_samples, 2000, replace=False)
                    true_vals = true_vals[indices]
                    residuals = residuals[indices]
                
                if param_name in self.log_scale_params:
                    # === SYMMETRIC LOG SCALE for LOG PARAMETERS ===
                    
                    epsilon = 1e-3
                    residuals_symlog = np.sign(residuals) * np.log10(np.abs(residuals) + epsilon)
                    
                    scatter = ax.scatter(true_vals, residuals_symlog, alpha=0.5, s=10, c='blue', edgecolors='none')
                    
                    # Zero line
                    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect prediction')
                    
                    # ±1σ reference lines
                    ref_1sigma_pos = np.log10(1.0 + epsilon)
                    ref_1sigma_neg = -np.log10(1.0 + epsilon)
                    ax.axhline(ref_1sigma_pos, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label='±1σ reference')
                    ax.axhline(ref_1sigma_neg, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                    
                    # === DYNAMIC Y-AXIS TICKS (same logic as above) ===
                    y_min, y_max = residuals_symlog.min(), residuals_symlog.max()
                    y_range = y_max - y_min
                    
                    # Create candidate tick values
                    candidate_values = []
                    for exp in range(-3, 3):
                        base = 10.0 ** exp
                        candidate_values.extend([
                            -10 * base, -3 * base, -base, -0.3 * base, -0.1 * base,
                            0.1 * base, 0.3 * base, base, 3 * base, 10 * base
                        ])
                    candidate_values.append(0.0)
                    candidate_values = np.array(sorted(set(candidate_values)))
                    
                    tick_positions = np.sign(candidate_values) * np.log10(np.abs(candidate_values) + epsilon)
                    
                    margin = y_range * 0.1
                    visible_mask = (tick_positions >= y_min - margin) & (tick_positions <= y_max + margin)
                    
                    filtered_positions = tick_positions[visible_mask]
                    filtered_values = candidate_values[visible_mask]
                    
                    # Subsample if too many
                    if len(filtered_positions) > 10:
                        stride = len(filtered_positions) // 8
                        indices = np.arange(0, len(filtered_positions), stride)
                        filtered_positions = filtered_positions[indices]
                        filtered_values = filtered_values[indices]
                    
                    # Ensure zero is included
                    if y_min < 0 < y_max:
                        zero_symlog = 0.0
                        if zero_symlog not in filtered_positions:
                            insert_idx = np.searchsorted(filtered_positions, zero_symlog)
                            filtered_positions = np.insert(filtered_positions, insert_idx, zero_symlog)
                            filtered_values = np.insert(filtered_values, insert_idx, 0.0)
                    
                    ax.set_yticks(filtered_positions)
                    
                    labels = []
                    for v in filtered_values:
                        if v == 0:
                            labels.append('0')
                        elif abs(v) >= 1:
                            labels.append(f'{v:+.0f}')
                        elif abs(v) >= 0.1:
                            labels.append(f'{v:+.1f}')
                        elif abs(v) >= 0.01:
                            labels.append(f'{v:+.2f}')
                        else:
                            labels.append(f'{v:+.3f}')
                    ax.set_yticklabels(labels, fontsize=8)
                    
                    ax.set_ylim(y_min - margin, y_max + margin)
                    
                    # Log scale for x-axis
                    ax.set_xscale('log')
                    
                    ax.set_xlabel(f'True {param_name} (log scale)', fontsize=10)
                    ax.set_ylabel(f'Residual: Δlog({param_name})\n[symlog scale]', fontsize=10)
                    ax.set_title(f'{param_name} (log-parameter)', fontsize=11, fontweight='bold')
                    
                    textstr = 'Y: sign(Δ)×log₁₀(|Δ|+ε)\nΔ = log(pred)−log(true)'
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                else:
                    # === STANDARD LINEAR PLOT ===
                    
                    scatter = ax.scatter(true_vals, residuals, alpha=0.5, s=10, c='blue', edgecolors='none')
                    
                    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect prediction')
                    
                    residual_std = np.std(residuals)
                    ax.axhline(residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'±1σ = ±{residual_std:.3f}')
                    ax.axhline(-residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                    
                    ax.set_xlabel(f'True {param_name}', fontsize=10)
                    ax.set_ylabel(f'Residual: pred - true', fontsize=10)
                    ax.set_title(f'{param_name} (linear parameter)', fontsize=11, fontweight='bold')
                
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)
            
            for i in range(n_params, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Epoch {epoch+1}: Residuals vs True Values (Symlog for Log-Params)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_residuals_vs_true.png'), 
                    dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_residuals_vs_true_symlog(self, results, epoch):
        """
        Plot residuals vs true values.
        
        CRITICAL FIX: For log-parameters, residuals are ALREADY Δlog(X), 
        so plot them on a LINEAR scale, not symlog!
        """
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            true_vals = results[param_name]['true_values_orig']
            residuals = results[param_name]['residuals_orig']
            
            # Subsample if too many points
            n_samples = len(true_vals)
            if n_samples > 2000:
                indices = np.random.choice(n_samples, 2000, replace=False)
                true_vals = true_vals[indices]
                residuals = residuals[indices]
            
            if param_name in self.log_scale_params:
                # === LOG PARAMETERS: Residuals are ALREADY Δlog(X) ===
                # No transformation needed! Just plot directly.
                
                scatter = ax.scatter(true_vals, residuals, alpha=0.5, s=10, c='blue', edgecolors='none')
                
                # Zero line
                ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect prediction')
                
                # ±1σ reference lines (in log-space)
                # For well-calibrated predictions, ~68% should be within ±1
                ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label='±1 in log-space')
                ax.axhline(-1.0, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                
                # Additional reference lines
                ax.axhline(0.3, color='orange', linestyle=':', linewidth=1, alpha=0.3)
                ax.axhline(-0.3, color='orange', linestyle=':', linewidth=1, alpha=0.3)
                
                # Log scale for x-axis (true values in physical space)
                ax.set_xscale('log')
                
                # LINEAR scale for y-axis (residuals already in log-space)
                ax.set_xlabel(f'True {param_name} (physical scale)', fontsize=10)
                ax.set_ylabel(f'Residual: log₁₀(pred) - log₁₀(true)', fontsize=10)
                ax.set_title(f'{param_name} (log-parameter)', fontsize=11, fontweight='bold')
                
                # Add explanation
                textstr = f'Y-axis: Δlog₁₀({param_name})\n(already in log-space)'
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            else:
                # === LINEAR PARAMETERS ===
                
                scatter = ax.scatter(true_vals, residuals, alpha=0.5, s=10, c='blue', edgecolors='none')
                
                ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect prediction')
                
                residual_std = np.std(residuals)
                ax.axhline(residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'±1σ = ±{residual_std:.3f}')
                ax.axhline(-residual_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
                
                ax.set_xlabel(f'True {param_name}', fontsize=10)
                ax.set_ylabel(f'Residual: pred - true', fontsize=10)
                ax.set_title(f'{param_name} (linear parameter)', fontsize=11, fontweight='bold')
            
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Residuals vs True Values', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_residuals_vs_true.png'), 
                dpi=150, bbox_inches='tight')
        plt.close()


    def _plot_residuals_vs_sigma_by_snr(self, results, epoch, snr):
        """
        Plot residuals vs predicted sigma, colored by SNR.
        
        CRITICAL FIX: For log-parameters, residuals are ALREADY Δlog(X),
        so plot them on a LINEAR scale, not symlog!
        """
        n_params = len(self.model_params)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        vmin, vmax = 10, 50
        norm = Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))
        cmap = plt.cm.viridis
        
        for i, param_name in enumerate(self.model_params):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            residuals = results[param_name]['residuals_orig']
            sigma_vals = results[param_name]['sigma_orig']
            # --- START: FIX ---
            # Filter out NaN residuals that may have been created for log-params with zero true values
            valid_mask = ~np.isnan(residuals)
            
            snr_log = np.log10(np.clip(snr[valid_mask], vmin, vmax))
            
            scatter = ax.scatter(sigma_vals[valid_mask], residuals[valid_mask], c=snr_log, cmap=cmap,
                               alpha=0.5, s=20, norm=norm)
            # --- END: FIX ---

            ax.axhline(0, color='red', linestyle='--', alpha=0.7)
            
            # --- START: MODIFIED AXIS SCALING AND LABELING ---
            if param_name in self.log_scale_params:
                # For log-params, sigma_orig can span orders of magnitude, so log scale is better.
                ax.set_xscale('log')
                # The y-axis is now relative error, which is symmetric around zero, so it should be linear.
                ax.set_ylabel('Relative Residual ((pred-true)/true)')
                # Use a symmetric log scale for y-axis if residuals span many orders of magnitude
                # For now, a linear scale is usually best for relative error.
            else:
                # For linear-params, a linear scale is usually fine for both.
                ax.set_ylabel('Residual (pred - true)')
            # --- END: MODIFIED AXIS SCALING AND LABELING ---

            ax.set_xlabel(f'Predicted σ')
            ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                cbar = plt.colorbar(scatter, ax=ax, label='SNR')
                cbar_ticks = [10, 15, 20, 30, 50]
                cbar.set_ticks(np.log10(cbar_ticks))
                cbar.set_ticklabels(cbar_ticks)
        
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}: Residuals vs Sigma (by SNR)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'E{epoch+1}_residuals_vs_sigma_snr.png'), dpi=150, bbox_inches='tight')
        plt.close()

    

    