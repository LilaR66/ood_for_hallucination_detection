#!/usr/bin/env python3


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
from src.analysis.evaluation import compute_metrics




def bootstrap_sample_indices(n_samples: int, n_bootstrap: int = 1000) -> List[np.ndarray]:
    """
    Generate bootstrap sample indices for resampling with replacement.
    
    This function creates multiple bootstrap samples by randomly sampling indices
    with replacement from the original dataset. Each bootstrap sample has the same
    size as the original dataset but may contain duplicate indices.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in the original dataset
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples to generate
        
    Returns
    -------
    bootstrap_indices : List[np.ndarray]
        List of arrays containing bootstrap sample indices
    """
    bootstrap_indices = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_indices.append(indices)
    return bootstrap_indices

    


def bootstrap_analysis(
    id_fit_embeddings: torch.Tensor,
    id_test_embeddings: torch.Tensor,
    od_test_embeddings: torch.Tensor,
    n_fit_samples: List[int],
    n_test_samples: List[int],
    compute_ood_score_fn: callable,
    n_bootstrap: int = 100,
    **kwargs
) -> Dict:
    """
    Perform bootstrap analysis to evaluate OOD detection performance across different sample sizes.
    
    This function:
    - Samples different numbers of training and test samples using bootstrap resampling
    - Evaluates OOD detection performance for each sample size configuration
    - Computes statistics (mean, std) across bootstrap iterations
    - Provides insights into sample size requirements for stable performance
    
    The bootstrap approach helps estimate the variability in performance metrics
    and understand how sample size affects model reliability. This is particularly
    useful for determining optimal dataset sizes and confidence intervals.
    
    Parameters
    ----------
    id_fit_embeddings : torch.Tensor
        Embeddings from in-distribution training samples.
        Shape: [n_total_fit_samples, embedding_dim]
    id_test_embeddings : torch.Tensor
        Embeddings from in-distribution test samples.
        Shape: [n_total_id_test_samples, embedding_dim]
    od_test_embeddings : torch.Tensor
        Embeddings from out-of-distribution test samples.
        Shape: [n_total_ood_test_samples, embedding_dim]
    n_fit_samples : List[int]
        List of training sample sizes to evaluate
    n_test_samples : List[int]
        List of test sample sizes to evaluate
    compute_ood_score_fn : callable
        Function for OOD detection. Must accept:
        - id_fit_embeddings, id_test_embeddings, od_test_embeddings
        - Additional kwargs for method-specific parameters
        Must return: (scores_id, scores_ood)
    n_bootstrap : int, optional (default=100)
        Number of bootstrap iterations to perform for each combination of training and 
        test sample sizes. Used to estimate the statistical variability (mean, std) 
        of performance metrics and plot confidence intervals. 
    **kwargs
        Additional keyword arguments passed to evaluate_ood_performance_fn
        (e.g., k=5 for DKNN, batch_size=1000)
        
    Returns
    -------
    results : Dict
        Dictionary containing bootstrap analysis results with keys:
        - 'n_fit_samples': List of training sample sizes tested
        - 'n_test_samples': List of test sample sizes tested  
        - 'auroc_mean': List of mean AUROC values across configurations
        - 'auroc_std': List of AUROC standard deviations across configurations
        - 'auprc_mean': List of mean AUPRC values across configurations
        - 'auprc_std': List of AUPRC standard deviations across configurations
        - 'fpr95_mean': List of mean FPR95 values across configurations
        - 'fpr95_std': List of FPR95 standard deviations across configurations
    """
    print("Starting bootstrap analysis...")
    
    results = {
        'n_fit_samples': n_fit_samples,
        'n_test_samples': n_test_samples,
        'auroc_mean': [],
        'auroc_std': [],
        'auprc_mean': [],
        'auprc_std': [],
        'fpr95_mean': [],
        'fpr95_std': []
    }
    
    for n_fit in tqdm(n_fit_samples, desc="Fit samples"):
        for n_test in tqdm(n_test_samples, desc="Test samples", leave=False):
            
            auroc_scores = []
            aucpr_scores = []
            fpr95_scores = []
            
            for _ in range(n_bootstrap):
                # Sample fit data
                fit_indices = np.random.choice(len(id_fit_embeddings), size=min(n_fit, len(id_fit_embeddings)), replace=False)
                fit_sample = id_fit_embeddings[fit_indices]
                
                # Sample test data
                id_test_indices = np.random.choice(len(id_test_embeddings), size=min(n_test, len(id_test_embeddings)), replace=False)
                od_test_indices = np.random.choice(len(od_test_embeddings), size=min(n_test, len(od_test_embeddings)), replace=False)
                
                id_test_sample = id_test_embeddings[id_test_indices]
                od_test_sample = od_test_embeddings[od_test_indices]
                
                try:
                    # Perform OOD detection: compute OOD scores
                    scores_id, scores_ood = compute_ood_score_fn(
                        fit_sample, id_test_sample, od_test_sample, **kwargs
                    )
                    # Compute metrics
                    auroc, fpr95, aucpr, _ ,_ ,_ ,_ = compute_metrics(scores_id, scores_ood, plot=False)
                    
                    auroc_scores.append(auroc)
                    aucpr_scores.append(aucpr)
                    fpr95_scores.append(fpr95)
                    
                except Exception as e:
                    print(f"Error with n_fit={n_fit}, n_test={n_test}: {e}")
                    continue
            
            # Store results
            results['auroc_mean'].append(np.mean(auroc_scores))
            results['auroc_std'].append(np.std(auroc_scores))
            results['auprc_mean'].append(np.mean(aucpr_scores))
            results['auprc_std'].append(np.std(aucpr_scores))
            results['fpr95_mean'].append(np.mean(fpr95_scores))
            results['fpr95_std'].append(np.std(fpr95_scores))
    
    return results




def plot_bootstrap_results(bootstrap_results: Dict, save_path: str = None):
    """
    Plot bootstrap analysis results showing performance vs sample size relationships.
    
    This function creates a comprehensive visualization of bootstrap analysis results
    including heatmaps of performance metrics and sample efficiency curves. The plots
    help identify optimal sample sizes and understand the trade-offs between
    training set size, test set size, and performance stability.
    
    Parameters
    ----------
    bootstrap_results : Dict
        Dictionary containing bootstrap analysis results from bootstrap_analysis().
        Must contain keys: 'n_fit_samples', 'n_test_samples', 'auroc_mean', 
        'auroc_std', 'auprc_mean', 'auprc_std', 'fpr95_mean', 'fpr95_std'
    save_path : str, optional (default=None)
        If provided, save the plot to this file path (e.g., "bootstrap_results.png")
        
    Returns
    -------
    None
        Displays the matplotlib figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Reshape results for plotting
    n_fit_samples = bootstrap_results['n_fit_samples']
    n_test_samples = bootstrap_results['n_test_samples']
    
    # Create meshgrid for heatmaps
    X, Y = np.meshgrid(n_test_samples, n_fit_samples)
    
    # Reshape metrics
    auroc_mean = np.array(bootstrap_results['auroc_mean']).reshape(len(n_fit_samples), len(n_test_samples))
    auroc_std = np.array(bootstrap_results['auroc_std']).reshape(len(n_fit_samples), len(n_test_samples))
    fpr95_mean = np.array(bootstrap_results['fpr95_mean']).reshape(len(n_fit_samples), len(n_test_samples))
    
    # Plot AUROC mean
    im1 = axes[0, 0].contourf(X, Y, auroc_mean, levels=20, cmap='viridis')
    axes[0, 0].set_xlabel('N Test Samples')
    axes[0, 0].set_ylabel('N Fit Samples')
    axes[0, 0].set_title('AUROC Mean')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot AUROC std
    im2 = axes[0, 1].contourf(X, Y, auroc_std, levels=20, cmap='viridis')
    axes[0, 1].set_xlabel('N Test Samples')
    axes[0, 1].set_ylabel('N Fit Samples')
    axes[0, 1].set_title('AUROC Std')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot FPR95 mean
    im3 = axes[1, 0].contourf(X, Y, fpr95_mean, levels=20, cmap='plasma') #coolwarm
    axes[1, 0].set_xlabel('N Test Samples')
    axes[1, 0].set_ylabel('N Fit Samples')
    axes[1, 0].set_title('FPR95 Mean')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot sample efficiency curve
    # Fix n_test and vary n_fit
    fixed_test_idx = len(n_test_samples) // 2
    auroc_vs_fit = auroc_mean[:, fixed_test_idx]
    auroc_std_vs_fit = auroc_std[:, fixed_test_idx]
    
    axes[1, 1].errorbar(n_fit_samples, auroc_vs_fit, yerr=auroc_std_vs_fit, 
                       marker='o', capsize=5, capthick=2)
    axes[1, 1].set_xlabel('N Fit Samples')
    axes[1, 1].set_ylabel('AUROC')
    axes[1, 1].set_title(f'Sample Efficiency (N Test = {n_test_samples[fixed_test_idx]})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




def display_bootstrap_statistics(
    bootstrap_results: dict,
    n_fit_samples_range: list,
    n_test_samples_range: list
):
    """
    Display summary statistics and recommendations from bootstrap analysis.

    Parameters
    ----------
    bootstrap_results : dict
        Dictionary containing keys 'auroc_mean', 'auroc_std', 'auprc_mean', 'auprc_std', 'fpr95_mean', 'fpr95_std'.
        Each value should be a list or np.ndarray of metric values for each configuration.
    n_fit_samples_range : list
        List of numbers of samples used for fitting in each configuration (outer loop).
    n_test_samples_range : list
        List of numbers of samples used for testing in each configuration (inner loop).

    Returns
    -------
    None
        This function prints the results to standard output.
    """
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)

    print(f"\nBootstrap Analysis (averages over {len(n_fit_samples_range)}×{len(n_test_samples_range)} configurations):")
    print(f"AUROC: {np.mean(bootstrap_results['auroc_mean']):.3f} ± {np.mean(bootstrap_results['auroc_std']):.3f}")
    print(f"AUPRC: {np.mean(bootstrap_results['auprc_mean']):.3f} ± {np.mean(bootstrap_results['auprc_std']):.3f}")
    print(f"FPR95: {np.mean(bootstrap_results['fpr95_mean']):.3f} ± {np.mean(bootstrap_results['fpr95_std']):.3f}")

    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)

    # Find the optimal number of samples for fitting
    best_config_idx = np.argmax(bootstrap_results['auroc_mean'])
    best_fit_idx = best_config_idx // len(n_test_samples_range)
    best_test_idx = best_config_idx % len(n_test_samples_range)

    print(f"\nOptimal configuration found:")
    print(f"  - Fit samples: {n_fit_samples_range[best_fit_idx]}")
    print(f"  - Test samples: {n_test_samples_range[best_test_idx]}")
    print(f"  - AUROC: {bootstrap_results['auroc_mean'][best_config_idx]:.3f} ± {bootstrap_results['auroc_std'][best_config_idx]:.3f}")

    # Analyze saturation effect
    print(f"\nSaturation analysis:")
    # AUROC for the smallest number of fit samples
    first_fit_aurocs = [bootstrap_results['auroc_mean'][i] for i in range(0, len(bootstrap_results['auroc_mean']), len(n_test_samples_range))]
    print(f"  - With {n_fit_samples_range[0]} fit samples: AUROC = {np.mean(first_fit_aurocs):.3f}")
    # AUROC for the largest number of fit samples
    last_fit_aurocs = [bootstrap_results['auroc_mean'][i] for i in range(len(n_test_samples_range)-1, len(bootstrap_results['auroc_mean']), len(n_test_samples_range))]
    print(f"  - With {n_fit_samples_range[-1]} fit samples: AUROC = {np.mean(last_fit_aurocs):.3f}")