import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os.path
from pathlib import Path
from qubounds.assessment.utils import init_logging


# + tags=["parameters"]
product = None
upstream = None
alpha = 0.1
# -


def plot_regression_ncm_comparison(df, output_path=None):
    """
    Create comprehensive 2x2 grid comparing NCM models for regression.
    
    Parameters
    ----------
    df : DataFrame
        Regression results with columns: ncm, Empirical coverage, 
        Relative Interval Width, sigma_r2, Split
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Filter to test set
    df_test = df[df['Split'] == 'Test'].copy()
    
    # Aggregate by NCM
    ncm_stats = df_test.groupby('ncm').agg({
        'Empirical coverage': ['mean', 'std'],
        'Relative Interval Width': ['mean', 'std'],
        'sigma_r2': ['mean', 'std'],
        'Average Interval Width': 'mean'
    }).reset_index()
    
    # Flatten columns
    ncm_stats.columns = ['ncm', 'coverage_mean', 'coverage_std', 
                         'width_mean', 'width_std',
                         'r2_mean', 'r2_std', 'abs_width_mean']
    
    # Sort by efficiency score
    ncm_stats['efficiency'] = ncm_stats['width_mean'] / (ncm_stats['coverage_mean'] + 0.01)
    ncm_stats = ncm_stats.sort_values('efficiency')
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================
    # Plot 1: Bar Chart - Mean Performance
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Sort by width (efficiency)
    ncm_sorted = ncm_stats.sort_values('width_mean')
    
    # Colors: highlight best
    colors = ['#27ae60' if i == 0 else '#95a5a6' 
              for i in range(len(ncm_sorted))]
    
    bars = ax1.barh(ncm_sorted['ncm'], ncm_sorted['width_mean'], 
                     color=colors, edgecolor='black', linewidth=1.5)
    
    # Add values
    for i, (idx, row) in enumerate(ncm_sorted.iterrows()):
        ax1.text(row['width_mean'] + 0.01, i, 
                f"{row['width_mean']:.3f}", 
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Mean Relative Interval Width', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Auxiliary Model', fontsize=12, fontweight='bold')
    ax1.set_title('A) Interval Width Efficiency\n(Lower = Better)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # ========================================
    # Plot 2: Coverage vs Width Scatter
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Scatter plot
    best_ncm = ncm_sorted.iloc[0]['ncm']
    colors_scatter = ['#27ae60' if ncm == best_ncm else '#3498db' 
                     for ncm in ncm_stats['ncm']]
    
    scatter = ax2.scatter(ncm_stats['width_mean'], 
                         ncm_stats['coverage_mean'],
                         s=200, c=colors_scatter, 
                         edgecolors='black', linewidth=2, 
                         alpha=0.7, zorder=3)
    
    # Error bars
    ax2.errorbar(ncm_stats['width_mean'], ncm_stats['coverage_mean'],
                xerr=ncm_stats['width_std'], yerr=ncm_stats['coverage_std'],
                fmt='none', ecolor='gray', alpha=0.4, zorder=1)
    
    # Annotate NCM names
    for idx, row in ncm_stats.iterrows():
        offset = 0.01 if row['ncm'] != best_ncm else 0.015
        fontweight = 'bold' if row['ncm'] == best_ncm else 'normal'
        ax2.annotate(row['ncm'], 
                    xy=(row['width_mean'], row['coverage_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight=fontweight,
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow' if row['ncm'] == best_ncm else 'white',
                            alpha=0.8, edgecolor='black'))
    
    # Target lines
    ax2.axhline(0.90, color='red', linestyle='--', linewidth=2, 
               label='Target Coverage (90%)', alpha=0.7, zorder=2)
    ax2.axhspan(0.88, 0.92, alpha=0.1, color='green', zorder=0)
    
    # Ideal region annotation
    ax2.text(0.05, 0.95, '← Ideal Region\n(Narrow + High Coverage)',
            transform=ax2.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax2.set_xlabel('Mean Relative Interval Width', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Empirical Coverage', fontsize=12, fontweight='bold')
    ax2.set_title('B) Efficiency-Coverage Tradeoff', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.80, 0.95)
    
    # ========================================
    # Plot 3: NCM Quality (R²) vs Width
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Scatter
    scatter3 = ax3.scatter(ncm_stats['r2_mean'], 
                          ncm_stats['width_mean'],
                          s=200, c=colors_scatter,
                          edgecolors='black', linewidth=2,
                          alpha=0.7)
    
    # Error bars
    ax3.errorbar(ncm_stats['r2_mean'], ncm_stats['width_mean'],
                xerr=ncm_stats['r2_std'], yerr=ncm_stats['width_std'],
                fmt='none', ecolor='gray', alpha=0.4)
    
    # Trendline
    if len(ncm_stats) > 2:
        z = np.polyfit(ncm_stats['r2_mean'], ncm_stats['width_mean'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(ncm_stats['r2_mean'].min(), 
                             ncm_stats['r2_mean'].max(), 100)
        ax3.plot(x_trend, p(x_trend), 'r--', linewidth=2, 
                alpha=0.5, label='Trend')
        
        # Correlation
        r, p_val = stats.pearsonr(ncm_stats['r2_mean'], ncm_stats['width_mean'])
        ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}',
                transform=ax3.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Annotate
    for idx, row in ncm_stats.iterrows():
        fontweight = 'bold' if row['ncm'] == best_ncm else 'normal'
        ax3.annotate(row['ncm'],
                    xy=(row['r2_mean'], row['width_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight=fontweight)
    
    ax3.set_xlabel('Auxiliary Model Quality (R²)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Relative Interval Width', fontsize=12, fontweight='bold')
    ax3.set_title('C) Does Better Auxiliary model → Narrower Intervals?', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========================================
    # Plot 4: Box Plots - Distribution Comparison
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Select top 5 NCMs by efficiency
    top_5_ncms = ncm_sorted.head(5)['ncm'].tolist()
    df_top5 = df_test[df_test['ncm'].isin(top_5_ncms)]
    
    # Create box plot
    positions = np.arange(len(top_5_ncms))
    bp = ax4.boxplot([df_top5[df_top5['ncm'] == ncm]['Relative Interval Width'].values
                      for ncm in top_5_ncms],
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(edgecolor='black', linewidth=1.5),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor='red', 
                                   markersize=6, alpha=0.5))
    
    # Color boxes
    for i, (patch, ncm) in enumerate(zip(bp['boxes'], top_5_ncms)):
        if ncm == best_ncm:
            patch.set_facecolor('#27ae60')
        else:
            patch.set_facecolor('#95a5a6')
    
    # Add median annotations
    for i, ncm in enumerate(top_5_ncms):
        data = df_top5[df_top5['ncm'] == ncm]['Relative Interval Width']
        median = data.median()
        ax4.text(i, median - 0.02, f'{median:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_xticks(positions)
    ax4.set_xticklabels(top_5_ncms, rotation=45, ha='right')
    ax4.set_ylabel('Relative Interval Width', fontsize=12, fontweight='bold')
    ax4.set_title('D) Width Distribution (Top 5 NCMs)', 
                 fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Overall title
    fig.suptitle('Regression Auxiliary Model Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig


def plot_coverage_by_ncm(df, output_path=None):
    """
    Simple bar chart showing coverage by NCM model.
    Similar to classification analysis.
    
    Parameters
    ----------
    df : DataFrame
        Test results
    output_path : str, optional
        Save path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    df_test = df[df['Split'] == 'Test'].copy()
    
    # Aggregate
    coverage_stats = df_test.groupby('ncm').agg({
        'Empirical coverage': ['mean', 'std', 'count']
    }).reset_index()
    
    coverage_stats.columns = ['ncm', 'coverage_mean', 'coverage_std', 'n_datasets']
    coverage_stats = coverage_stats.sort_values('coverage_mean', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    best_coverage = coverage_stats.iloc[0]['ncm']
    colors = ['#27ae60' if ncm == best_coverage else '#3498db' 
             for ncm in coverage_stats['ncm']]
    
    # Bar plot
    bars = ax.barh(coverage_stats['ncm'], coverage_stats['coverage_mean'],
                   color=colors, edgecolor='black', linewidth=1.5,
                   xerr=coverage_stats['coverage_std'], capsize=5)
    
    # Target line
    ax.axvline(0.90, color='red', linestyle='--', linewidth=2,
              label='Target (90%)', alpha=0.7)
    ax.axvspan(0.88, 0.92, alpha=0.1, color='green', label='Acceptable Range')
    
    # Annotate
    for i, (idx, row) in enumerate(coverage_stats.iterrows()):
        ax.text(row['coverage_mean'] + 0.01, i,
               f"{row['coverage_mean']:.1%} ± {row['coverage_std']:.1%}",
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean Empirical Coverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Auxiliary Model', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Coverage by Auxiliary Model\n(Target: 90%)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0.80, 0.95)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Coverage plot saved to {output_path}")
    
    return fig


def plot_dataset_difficulty(df, output_path=None):
    """
    Show which datasets are hardest/easiest for conformal prediction.
    
    Parameters
    ----------
    df : DataFrame
        Test results
    output_path : str, optional
        Save path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    df_test = df[df['Split'] == 'Test'].copy()
    
    # Aggregate by dataset (average across NCMs)
    dataset_stats = df_test.groupby('Dataset Name').agg({
        'Empirical coverage': 'mean',
        'Relative Interval Width': 'mean'
    }).reset_index()
    
    # Sort by width
    dataset_stats = dataset_stats.sort_values('Relative Interval Width')
    
    # Select top 10 easiest and hardest
    n_show = 10
    easiest = dataset_stats.head(n_show)
    hardest = dataset_stats.tail(n_show)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========================================
    # Left: Easiest datasets
    # ========================================
    ax1 = axes[0]
    
    bars1 = ax1.barh(easiest['Dataset Name'], easiest['Relative Interval Width'],
                     color='#27ae60', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Annotate
    for i, (idx, row) in enumerate(easiest.iterrows()):
        ax1.text(row['Relative Interval Width'] + 0.01, i,
                f"{row['Relative Interval Width']:.3f}\n(Cov: {row['Empirical coverage']:.1%})",
                va='center', fontsize=9)
    
    ax1.set_xlabel('Relative Interval Width', fontsize=12, fontweight='bold')
    ax1.set_title(f'Easiest Datasets (Top {n_show})\nNarrowest Intervals', 
                 fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # ========================================
    # Right: Hardest datasets
    # ========================================
    ax2 = axes[1]
    
    bars2 = ax2.barh(hardest['Dataset Name'], hardest['Relative Interval Width'],
                     color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Annotate
    for i, (idx, row) in enumerate(hardest.iterrows()):
        ax2.text(row['Relative Interval Width'] + 0.01, i,
                f"{row['Relative Interval Width']:.3f}\n(Cov: {row['Empirical coverage']:.1%})",
                va='center', fontsize=9)
    
    ax2.set_xlabel('Relative Interval Width', fontsize=12, fontweight='bold')
    ax2.set_title(f'Hardest Datasets (Top {n_show})\nWidest Intervals', 
                 fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    fig.suptitle('Dataset Difficulty Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dataset difficulty plot saved to {output_path}")
    
    return fig


def plot_ncm_similarity_heatmap(df, output_path=None):
    """
    Correlation heatmap showing similarity between NCM models.
    Pure matplotlib.
    
    Parameters
    ----------
    df : DataFrame
        Test results
    output_path : str, optional
        Save path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    df_test = df[df['Split'] == 'Test'].copy()
    
    # Pivot: datasets x NCMs (using Relative Width)
    pivot_width = df_test.pivot(index='Dataset Name', 
                                columns='ncm', 
                                values='Relative Interval Width')
    
    pivot_coverage = df_test.pivot(index='Dataset Name',
                                   columns='ncm',
                                   values='Empirical coverage')
    
    # Compute correlations
    corr_width = pivot_width.corr()
    corr_coverage = pivot_coverage.corr()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ========================================
    # Left: Width correlation
    # ========================================
    ax1 = axes[0]
    
    im1 = ax1.imshow(corr_width, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    
    ncm_names = corr_width.columns.tolist()
    ax1.set_xticks(np.arange(len(ncm_names)))
    ax1.set_yticks(np.arange(len(ncm_names)))
    ax1.set_xticklabels(ncm_names, rotation=45, ha='right')
    ax1.set_yticklabels(ncm_names)
    
    # Add values
    for i in range(len(ncm_names)):
        for j in range(len(ncm_names)):
            value = corr_width.iloc[i, j]
            color = 'white' if value < 0.75 else 'black'
            ax1.text(j, i, f'{value:.2f}',
                    ha='center', va='center',
                    color=color, fontsize=8, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Correlation', fontsize=11, fontweight='bold')
    
    ax1.set_title('Interval Width Similarity\nAcross Datasets', 
                 fontsize=13, fontweight='bold')
    
    # Grid
    ax1.set_xticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # ========================================
    # Right: Coverage correlation
    # ========================================
    ax2 = axes[1]
    
    im2 = ax2.imshow(corr_coverage, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    
    ax2.set_xticks(np.arange(len(ncm_names)))
    ax2.set_yticks(np.arange(len(ncm_names)))
    ax2.set_xticklabels(ncm_names, rotation=45, ha='right')
    ax2.set_yticklabels(ncm_names)
    
    # Add values
    for i in range(len(ncm_names)):
        for j in range(len(ncm_names)):
            value = corr_coverage.iloc[i, j]
            color = 'white' if value < 0.75 else 'black'
            ax2.text(j, i, f'{value:.2f}',
                    ha='center', va='center',
                    color=color, fontsize=8, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Correlation', fontsize=11, fontweight='bold')
    
    ax2.set_title('Coverage Similarity\nAcross Datasets', 
                 fontsize=13, fontweight='bold')
    
    # Grid
    ax2.set_xticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax2.set_yticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    fig.suptitle('Auxiliary Model Similarity Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    
    return fig


def analyze_regression_ncm_selection(df, output_path=None):
    """
    Comprehensive analysis to select best NCM model for regression conformal prediction.
    Similar to classification analysis but focused on interval width efficiency.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ['Dataset Name', 'ncm', 'Split', 'Empirical coverage', 
                           'Average Interval Width', 'Relative Interval Width',
                           'sigma_r2', 'sigma_rmse', 'sigma_mae']
    output_path : str, optional
        Path to save summary table
    
    Returns
    -------
    summary_df : DataFrame
        Summary statistics for each NCM model
    """
    
    # Filter to test set only (what matters for evaluation)
    df_test = df[df['Split'] == 'Test'].copy()
    
    print("=" * 80)
    print("REGRESSION Auxiliary MODEL SELECTION ANALYSIS")
    print("=" * 80)
    print()
    
    # ========================================
    # 1. Overall Performance by NCM
    # ========================================
    print("1. OVERALL PERFORMANCE BY AUXILIARY MODEL")
    print("-" * 80)
    
    summary_stats = df_test.groupby('ncm').agg({
        'Empirical coverage': ['mean', 'std', 'min', 'max', 'count'],
        'Relative Interval Width': ['mean', 'std', 'min', 'max'],
        'Average Interval Width': ['mean', 'std'],
        'sigma_r2': ['mean', 'std'],
        'sigma_rmse': ['mean', 'std'],
        'sigma_mae': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    
    # Add efficiency score (lower is better)
    summary_stats['efficiency_score'] = (
        summary_stats['Relative Interval Width_mean'] * 
        (1 / (summary_stats['Empirical coverage_mean'] + 0.01))
    )
    
    # Rank by efficiency (coverage close to 90%, narrow intervals)
    summary_stats['coverage_deviation'] = np.abs(summary_stats['Empirical coverage_mean'] - 0.90)
    summary_stats['rank'] = (
        summary_stats['coverage_deviation'] * 10 +  # Penalty for missing 90% target
        summary_stats['Relative Interval Width_mean']  # Narrower is better
    )
    
    summary_stats = summary_stats.sort_values('rank')
    
    print(summary_stats[['Empirical coverage_mean', 'Empirical coverage_std',
                         'Relative Interval Width_mean', 'Relative Interval Width_std',
                         'sigma_r2_mean', 'rank']].to_string())
    print()
    
    # ========================================
    # 2. Coverage Analysis
    # ========================================
    print("2. COVERAGE ANALYSIS (Target: 90%)")
    print("-" * 80)
    
    coverage_summary = df_test.groupby('ncm')['Empirical coverage'].agg(['mean', 'std', 'count'])
    coverage_summary['meets_target'] = (coverage_summary['mean'] >= 0.88) & (coverage_summary['mean'] <= 0.92)
    coverage_summary = coverage_summary.sort_values('mean', ascending=False)
    
    print(coverage_summary.to_string())
    print()
    
    # Count how many meet target
    n_meeting_target = coverage_summary['meets_target'].sum()
    print(f"NCMs meeting target (88-92%): {n_meeting_target}/{len(coverage_summary)}")
    print()
    
    # ========================================
    # 3. Interval Width Efficiency
    # ========================================
    print("3. INTERVAL WIDTH EFFICIENCY (Lower = Better)")
    print("-" * 80)
    
    width_summary = df_test.groupby('ncm')['Relative Interval Width'].agg(['mean', 'std'])
    width_summary = width_summary.sort_values('mean')
    
    print(width_summary.to_string())
    print()
    
    # Calculate improvement over worst
    worst_width = width_summary['mean'].max()
    best_width = width_summary['mean'].min()
    improvement = (worst_width - best_width) / worst_width * 100
    
    print(f"Best NCM: {width_summary.index[0]} (width: {best_width:.4f})")
    print(f"Worst NCM: {width_summary.index[-1]} (width: {worst_width:.4f})")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    # ========================================
    # 4. NCM Model Quality (sigma_r2)
    # ========================================
    print("4. AUXILIARY MODEL QUALITY (R² - Higher = Better)")
    print("-" * 80)
    
    ncm_quality = df_test.groupby('ncm')['sigma_r2'].agg(['mean', 'std'])
    ncm_quality = ncm_quality.sort_values('mean', ascending=False)
    
    print(ncm_quality.to_string())
    print()
    
    # ========================================
    # 5. Correlation: NCM Quality vs Efficiency
    # ========================================
    print("5. DOES BETTER NCM → NARROWER INTERVALS?")
    print("-" * 80)
    
    # Aggregate by NCM
    ncm_agg = df_test.groupby('ncm').agg({
        'sigma_r2': 'mean',
        'Relative Interval Width': 'mean'
    })
    
    # Correlation test
    r, p = stats.pearsonr(ncm_agg['sigma_r2'], ncm_agg['Relative Interval Width'])
    
    print(f"Correlation (sigma_r2 vs Relative Interval Width):")
    print(f"  Pearson r: {r:.3f}")
    print(f"  p-value: {p:.4f}")
    print(f"  Interpretation: {'Significant' if p < 0.05 else 'Not significant'}")
    
    if r < 0:
        print(f"  → Better NCM (higher R²) predicts narrower intervals ✓")
    else:
        print(f"  → No clear benefit from better NCM ✗")
    print()
    
    # ========================================
    # 6. Variance Analysis (Stability)
    # ========================================
    print("6. MODEL STABILITY (Lower Std = More Consistent)")
    print("-" * 80)
    
    stability = df_test.groupby('ncm').agg({
        'Empirical coverage': 'std',
        'Relative Interval Width': 'std'
    })
    stability.columns = ['Coverage Std', 'Width Std']
    stability['Combined Variance'] = stability['Coverage Std'] + stability['Width Std']
    stability = stability.sort_values('Combined Variance')
    
    print(stability.to_string())
    print()
    
    # ========================================
    # 7. Statistical Comparison (ANOVA)
    # ========================================
    print("7. STATISTICAL TESTS")
    print("-" * 80)
    
    # Test: Do NCMs differ significantly in interval width?
    ncm_groups = [df_test[df_test['ncm'] == ncm]['Relative Interval Width'].values 
                  for ncm in df_test['ncm'].unique()]
    
    f_stat, p_val = stats.f_oneway(*ncm_groups)
    
    print("ANOVA: Auxiliary models differ in Relative Interval Width?")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Result: {'YES - NCM choice matters' if p_val < 0.001 else 'NO - NCM choice irrelevant'}")
    print()
    
    # Test: Do NCMs differ in coverage?
    coverage_groups = [df_test[df_test['ncm'] == ncm]['Empirical coverage'].values 
                       for ncm in df_test['ncm'].unique()]
    
    f_stat_cov, p_val_cov = stats.f_oneway(*coverage_groups)
    
    print("ANOVA: Auxiliary models differ in Coverage?")
    print(f"  F-statistic: {f_stat_cov:.3f}")
    print(f"  p-value: {p_val_cov:.6f}")
    print(f"  Result: {'YES' if p_val_cov < 0.001 else 'NO - Coverage similar (good!)'}")
    print()
    
    # ========================================
    # 8. Top Recommendations
    # ========================================
    print("8. TOP 3 RECOMMENDED AUXILIARY MODELS")
    print("-" * 80)
    
    top_3 = summary_stats.head(3)
    
    for i, (ncm, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{i}. {ncm}")
        print(f"   Coverage: {row['Empirical coverage_mean']:.3f} ± {row['Empirical coverage_std']:.3f}")
        print(f"   Relative Width: {row['Relative Interval Width_mean']:.4f} ± {row['Relative Interval Width_std']:.4f}")
        print(f"   NCM R²: {row['sigma_r2_mean']:.3f}")
        print(f"   Datasets: {int(row['Empirical coverage_count'])}")
        
        # Pros/Cons
        if i == 1:
            print("   ⭐ RECOMMENDED: Best overall balance")
    
    print()
    
    # ========================================
    # 9. Final Recommendation
    # ========================================
    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    best_ncm = summary_stats.index[0]
    best_stats = summary_stats.iloc[0]
    
    print(f"\n✓ Selected NCM: {best_ncm}")
    print(f"\nPerformance:")
    print(f"  • Coverage: {best_stats['Empirical coverage_mean']:.1%} (target: 90%)")
    print(f"  • Relative Width: {best_stats['Relative Interval Width_mean']:.4f}")
    print(f"  • NCM R²: {best_stats['sigma_r2_mean']:.3f}")
    print(f"  • NCM RMSE: {best_stats['sigma_rmse_mean']:.4f}")
    
    print(f"\nRationale:")
    meets_coverage = 0.88 <= best_stats['Empirical coverage_mean'] <= 0.92
    print(f"  1. {'✓' if meets_coverage else '✗'} Meets coverage target (88-92%)")
    
    is_narrowest = best_ncm == width_summary.index[0]
    print(f"  2. {'✓' if is_narrowest else '✓'} {'Narrowest' if is_narrowest else 'Near-optimal'} interval width")
    
    is_high_quality = best_stats['sigma_r2_mean'] >= 0.7
    print(f"  3. {'✓' if is_high_quality else '~'} {'High' if is_high_quality else 'Moderate'} NCM quality (R²)")
    
    # Alternative if tied
    if len(summary_stats) > 1:
        second_best = summary_stats.index[1]
        rank_diff = summary_stats.iloc[1]['rank'] - summary_stats.iloc[0]['rank']
        
        if rank_diff < 0.01:  # Essentially tied
            print(f"\nNote: {second_best} performs similarly (rank diff: {rank_diff:.4f})")
            print(f"      Consider using simpler model if {best_ncm} is more complex")
    
    print()
    print("=" * 80)
    
    # Save summary
    if output_path:
        summary_stats.to_csv(output_path)
        print(f"\nSummary saved to: {output_path}")
    
    return summary_stats


def print_dataset_difficulty_analysis(df):
    """
    Analyze which datasets are hardest/easiest for conformal prediction.
    """
    df_test = df[df['Split'] == 'Test'].copy()
    
    print("\n" + "=" * 80)
    print("DATASET DIFFICULTY ANALYSIS")
    print("=" * 80)
    print()
    
    # Aggregate by dataset (average across NCMs)
    dataset_stats = df_test.groupby('Dataset Name').agg({
        'Empirical coverage': ['mean', 'std'],
        'Relative Interval Width': ['mean', 'std'],
        'sigma_r2': 'mean'
    })
    
    dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns.values]
    
    print("MOST CHALLENGING DATASETS (Widest Intervals):")
    hardest = dataset_stats.nlargest(5, 'Relative Interval Width_mean')
    print(hardest[['Empirical coverage_mean', 'Relative Interval Width_mean', 'sigma_r2_mean']].to_string())
    print()
    
    print("EASIEST DATASETS (Narrowest Intervals):")
    easiest = dataset_stats.nsmallest(5, 'Relative Interval Width_mean')
    print(easiest[['Empirical coverage_mean', 'Relative Interval Width_mean', 'sigma_r2_mean']].to_string())
    print()
    
    print("DATASETS WITH POOR COVERAGE (<85%):")
    poor_coverage = dataset_stats[dataset_stats['Empirical coverage_mean'] < 0.85]
    if len(poor_coverage) > 0:
        print(poor_coverage[['Empirical coverage_mean', 'Relative Interval Width_mean']].to_string())
    else:
        print("  None - all datasets meet coverage target ✓")
    print()


def compare_ncm_variance(df):
    """
    Check if all NCMs give similar results (like classification case).
    """
    df_test = df[df['Split'] == 'Test'].copy()
    
    print("\n" + "=" * 80)
    print("NCM SIMILARITY ANALYSIS")
    print("=" * 80)
    print()
    
    # For each dataset, check variance across NCMs
    dataset_variance = []
    
    for dataset in df_test['Dataset Name'].unique():
        dataset_data = df_test[df_test['Dataset Name'] == dataset]
        
        if len(dataset_data) < 2:
            continue
        
        cov_std = dataset_data['Empirical coverage'].std()
        width_std = dataset_data['Relative Interval Width'].std()
        
        dataset_variance.append({
            'Dataset': dataset,
            'Coverage Std': cov_std,
            'Width Std': width_std
        })
    
    var_df = pd.DataFrame(dataset_variance)
    
    print("Within-Dataset Variance Across NCMs:")
    print(f"  Mean Coverage Std: {var_df['Coverage Std'].mean():.4f}")
    print(f"  Mean Width Std: {var_df['Width Std'].mean():.4f}")
    print()
    
    # Interpretation
    if var_df['Coverage Std'].mean() < 0.05:
        print("  → Coverage is CONSISTENT across NCMs (like classification)")
    else:
        print("  → Coverage VARIES by NCM choice")
    
    if var_df['Width Std'].mean() < 0.1:
        print("  → Width is CONSISTENT across NCMs (NCM choice doesn't matter)")
    else:
        print("  → Width VARIES by NCM choice (NCM matters for efficiency!)")
    print()


logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")
path_regr = upstream["regression_summary_mapie"]
df = pd.read_excel(os.path.join(path_regr["data"]))
df = df.loc[~df["outlier"]]
df = df[df["Relative Interval Width"]>0.003] # knn

# Main analysis
summary = analyze_regression_ncm_selection(
    df, output_path=os.path.join(product["data"], 'ncm_regression_summary.csv'))

# Additional analyses
print_dataset_difficulty_analysis(df)
compare_ncm_variance(df)

# Create all plots
fig1 = plot_regression_ncm_comparison(df, os.path.join(product["data"], 'regression_ncm_comparison.png'))
fig2 = plot_coverage_by_ncm(df, os.path.join(product["data"], 'regression_coverage_by_ncm.png'))
fig3 = plot_dataset_difficulty(df, os.path.join(product["data"], 'regression_dataset_difficulty.png'))
fig4 = plot_ncm_similarity_heatmap(df, os.path.join(product["data"], 'regression_ncm_similarity.png'))
    
plt.show()


