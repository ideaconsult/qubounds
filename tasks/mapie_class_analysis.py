import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path


# + tags=["parameters"]
product = None
upstream = None
alpha = 0.1
# -


"""
NCM-Based Pseudo-Probabilistic Conformal Prediction

This approach bridges the gap between hard-prediction classifiers and probability-based conformal methods.
Since the external toxicity classifier only provides hard class predictions (e.g., "toxicity = 2") without
confidence scores, we cannot directly use standard MAPIE conformity scores like LAC (Least Ambiguous Classifier),
which require predict_proba(). To solve this, we train a separate Nonconformity Measure (NCM) model that learns
to predict the probability distribution over ordinal distances: P(distance = 0, 1, 2, 3 | molecule). During prediction,
we convert these distance probabilities into pseudo-class probabilities using the relationship: 
    P(class = j | molecule, ŷ) = P(distance = |j - ŷ| | molecule). 

For example, if the external model predicts class ŷ=1 and the NCM outputs [0.6, 0.3, 0.08, 0.02] for distances [0,1,2,3],
we first assign raw pseudo-probabilities:
    - P_raw(class=0) = P(distance=|0-1|=1) = 0.3
    - P_raw(class=1) = P(distance=|1-1|=0) = 0.6  ← predicted class
    - P_raw(class=2) = P(distance=|2-1|=1) = 0.3
    - P_raw(class=3) = P(distance=|3-1|=2) = 0.08

Since classes 0 and 2 are both distance 1 from the prediction, they share the probability mass P(distance=1)=0.3.
We normalize to ensure valid probabilities: sum = 0.6 + 0.3 + 0.3 + 0.08 = 1.28, giving final probabilities:
    - P(class=0) = 0.3/1.28 = 0.234
    - P(class=1) = 0.6/1.28 = 0.469  ← highest (predicted class)
    - P(class=2) = 0.3/1.28 = 0.234
    - P(class=3) = 0.08/1.28 = 0.063

These synthetic probabilities encode both the hard prediction (highest probability at predicted class) and uncertainty
(spread reflects NCM's confidence). We then feed these pseudo-probabilities into MAPIE's standard LAC conformity score,
which computes conformity as the difference between top predicted probability and each class's probability. This allows
us to leverage MAPIE's proven conformal framework while working with hard predictions, combining the ordinal structure
awareness of our NCM with the rigorous coverage guarantees of split conformal prediction.

NCM-Based Pseudo-Probabilistic Conformal Prediction for New Compounds

When predicting on a new, unseen compound:

Step 1: External Classifier Prediction
The pre-trained external toxicity classifier outputs a hard class prediction: ŷ_new (e.g., toxicity class = 2). 
This prediction is deterministic with no confidence score.

Step 2: NCM Distance Prediction
The trained NCM model takes the compound's ECFP fingerprint as input and predicts the probability distribution
over how far the external classifier's prediction is likely to be from the true class:
    NCM(ECFP_new) → [P(distance=0), P(distance=1), P(distance=2), P(distance=3)]
For example: [0.5, 0.35, 0.12, 0.03], indicating the model expects the prediction to be correct (distance=0)
with 50% probability, off by 1 class with 35% probability, etc.

Step 3: Convert to Class Pseudo-Probabilities
Given ŷ_new=2 and NCM output [0.5, 0.35, 0.12, 0.03], we compute pseudo-probabilities for each possible class:
    - P_raw(class=0) = P(distance=|0-2|=2) = 0.12
    - P_raw(class=1) = P(distance=|1-2|=1) = 0.35
    - P_raw(class=2) = P(distance=|2-2|=0) = 0.5   ← predicted class (highest)
    - P_raw(class=3) = P(distance=|3-2|=1) = 0.35

Classes 1 and 3 share the same probability because they're equidistant from the prediction (ordinal symmetry).
Normalizing: sum = 0.12 + 0.35 + 0.5 + 0.35 = 1.32, so:
    - P(class=0) = 0.12/1.32 = 0.091
    - P(class=1) = 0.35/1.32 = 0.265
    - P(class=2) = 0.5/1.32 = 0.379  ← highest
    - P(class=3) = 0.35/1.32 = 0.265

Step 4: LAC Conformity Score
MAPIE's LAC score computes, for each class j:
    score(j) = P(ŷ_new) - P(class=j) = P(class=2) - P(class=j)
    
    - score(class=0) = 0.379 - 0.091 = 0.288
    - score(class=1) = 0.379 - 0.265 = 0.114
    - score(class=2) = 0.379 - 0.379 = 0.000  ← predicted class (lowest score)
    - score(class=3) = 0.379 - 0.265 = 0.114

Step 5: Build Prediction Set
During calibration, a threshold τ was computed (e.g., τ = 0.15). The prediction set includes all classes where:
    score(j) ≤ τ

In this example:
    - class=0: 0.288 > 0.15 → excluded
    - class=1: 0.114 ≤ 0.15 → included
    - class=2: 0.000 ≤ 0.15 → included (predicted class always in set)
    - class=3: 0.114 ≤ 0.15 → included

Final prediction set: {1, 2, 3}

Interpretation: While the external classifier predicts class 2, the conformal system indicates the true toxicity
could reasonably be class 1, 2, or 3 with 90% confidence, reflecting the uncertainty captured by the NCM model.
If the NCM had been more confident (higher P(distance=0)), the set would be smaller (possibly just {2}). If less
confident (flatter distribution), the set would include class 0 as well.
"""

def plot_ncm_coverage_comparison(df, output_path=None):
    """
    Create comprehensive visualization comparing NCM model performance.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ['Dataset Name', 'ncm', 'Empirical Coverage', 'Split']
        where split is 'Training' or 'Calibration'
    output_path : str, optional
        Path to save figure (e.g., 'ncm_comparison.png')
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Filter to calibration only (what matters for conformal)
    df_cal = df[df['Split'] == 'Calibration'].copy()
    
    # Classify NCM types
    def classify_ncm(ncm_name):
        if ncm_name.startswith('c') or ncm_name.startswith('o'):
            return 'Classifier'
        else:
            return 'Regressor'
    
    df_cal['ncm_type'] = df_cal['ncm'].apply(classify_ncm)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================
    # Plot 1: Bar Chart - Mean Coverage by NCM
    # ========================================
    ax1 = axes[0]
    
    # Calculate mean coverage per NCM
    mean_coverage = df_cal.groupby(['ncm', 'ncm_type'])['Empirical Coverage'].mean().reset_index()
    mean_coverage = mean_coverage.sort_values('Empirical Coverage', ascending=False)
    
    # Highlight colors
    highlight_colors = []
    for idx, row in mean_coverage.iterrows():
        if row['ncm'] == 'crfecfp':
            highlight_colors.append('#27ae60')  # Dark green (recommended)
        elif row['ncm_type'] == 'Classifier':
            highlight_colors.append('#95a5a6')  # Gray
        else:
            highlight_colors.append('#e74c3c')  # Red (regressor)
    
    bars = ax1.barh(mean_coverage['ncm'], mean_coverage['Empirical Coverage'], 
                     color=highlight_colors, edgecolor='black', linewidth=1.5)
    
    # Add target line at 0.90
    ax1.axvline(1-alpha, color='blue', linestyle='--', linewidth=2, 
                label=f'Target ({100*(1-alpha)}%)', alpha=0.7)
    
    # Annotate values
    for i, (idx, row) in enumerate(mean_coverage.iterrows()):
        ax1.text(row['Empirical Coverage'] + 0.01, i, 
                f"{row['Empirical Coverage']:.2f}", 
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Mean Calibration Coverage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('NCM Model', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage Performance by NCM Model', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.75, 0.95)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add separator line between classifiers and regressors
    classifier_count = (mean_coverage['ncm_type'] == 'Classifier').sum()
    if classifier_count < len(mean_coverage):
        ax1.axhline(classifier_count - 0.5, color='black', 
                   linestyle='-', linewidth=2, alpha=0.5)
        ax1.text(0.76, classifier_count - 0.3, 'Classifiers', 
                fontsize=10, va='bottom', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.text(0.76, classifier_count - 0.7, 'Regressors', 
                fontsize=10, va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================
    # Plot 2: Box Plot - Coverage Distribution
    # ========================================
    ax2 = axes[1]
    
    # Select subset of NCMs to compare
    ncms_to_plot = ['crfecfp', 'cmlpecfp', 'cknnecfp', 'ladecfp', 'rfecfp', 'gbecfp']
    
    df_subset = df_cal[df_cal['ncm'].isin(ncms_to_plot)].copy()
    
    # Order by mean coverage
    ncm_order = df_subset.groupby('ncm')['Empirical Coverage'].mean().sort_values(ascending=False).index
    
    # Create box plot
    positions = np.arange(len(ncm_order))
    bp = ax2.boxplot([df_subset[df_subset['ncm'] == ncm]['Empirical Coverage'].values 
                       for ncm in ncm_order],
                      positions=positions,
                      widths=0.6,
                      patch_artist=True,
                      medianprops=dict(color='black', linewidth=2),
                      boxprops=dict(edgecolor='black', linewidth=1.5),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
    
    # Color boxes
    for i, (patch, ncm) in enumerate(zip(bp['boxes'], ncm_order)):
        if ncm == 'crfecfp':
            patch.set_facecolor('#27ae60')  # Green for recommended
        elif ncm in ['cmlpecfp', 'cknnecfp', 'ladecfp']:
            patch.set_facecolor('#95a5a6')  # Gray for other classifiers
        else:
            patch.set_facecolor('#e74c3c')  # Red for regressors
    
    # Target line
    ax2.axhline(1-alpha, color='blue', linestyle='--', linewidth=2, 
                label=f'Target ({100*(1-alpha)}%)', alpha=0.7)    
    
    ax2.set_ylabel('Calibration Coverage', fontsize=12, fontweight='bold')
    ax2.set_title('Coverage Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([ncm for ncm in ncm_order], rotation=45, ha='right')
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add median value annotations
    for i, ncm in enumerate(ncm_order):
        data = df_subset[df_subset['ncm'] == ncm]['Empirical Coverage']
        median = data.median()
        ax2.text(i, median + 0.02, f'{median:.2f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig


def plot_ncm_heatmap(df, output_path=None):
    """
    Create correlation heatmap showing similarity between NCM models.
    Pure matplotlib implementation without seaborn.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ['Dataset Name', 'ncm', 'Empirical Coverage', 'split']
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Filter to calibration only
    df_cal = df[df['Split'] == 'Calibration'].copy()
    
    # Pivot to wide format: rows=datasets, columns=NCM models
    pivot = df_cal.pivot(index='Dataset Name', columns='ncm', values='Empirical Coverage')
    
    # Compute correlation matrix
    corr_matrix = pivot.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormap
    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.85, 1.0
    
    # Plot heatmap using imshow
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Set ticks
    ncm_names = corr_matrix.columns.tolist()
    ax.set_xticks(np.arange(len(ncm_names)))
    ax.set_yticks(np.arange(len(ncm_names)))
    ax.set_xticklabels(ncm_names, rotation=45, ha='right')
    ax.set_yticklabels(ncm_names)
    
    # Add correlation values as text
    for i in range(len(ncm_names)):
        for j in range(len(ncm_names)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if value < 0.92 else 'black'
            ax.text(j, i, f'{value:.2f}', 
                   ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ncm_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    ax.set_title('NCM Model Similarity\n(Correlation of Calibration Coverage Across Datasets)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    
    return fig


def plot_classifier_vs_regressor_comparison(df, output_path=None):
    """
    Create side-by-side comparison of classifier vs regressor NCMs.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ['Dataset Name', 'ncm', 'Empirical Coverage', 'split']
    output_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    df_cal = df[df['Split'] == 'Calibration'].copy()
    
    # Classify NCM types
    df_cal['ncm_type'] = df_cal['ncm'].apply(
        lambda x: 'Classifier' if x.startswith(('c', 'o')) else 'Regressor'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data for each type
    classifiers = df_cal[df_cal['ncm_type'] == 'Classifier']['Empirical Coverage']
    regressors = df_cal[df_cal['ncm_type'] == 'Regressor']['Empirical Coverage']
    
    # Create violin plot-style using multiple box plots
    data_to_plot = [classifiers, regressors]
    positions = [1, 2]
    
    bp = ax.boxplot(data_to_plot, 
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=3),
                    boxprops=dict(edgecolor='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=2),
                    capprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6))
    
    # Color boxes
    bp['boxes'][0].set_facecolor('#27ae60')  # Green for classifiers
    bp['boxes'][1].set_facecolor('#e74c3c')  # Red for regressors
    
    # Add target line
    ax.axhline(1-alpha, color='blue', linestyle='--', linewidth=2, 
                label=f'Target ({100*(1-alpha)}%)', alpha=0.7)    
    
    # Add mean markers
    means = [classifiers.mean(), regressors.mean()]
    ax.plot(positions, means, 'D', color='gold', markersize=12, 
           markeredgecolor='black', markeredgewidth=2, label='Mean', zorder=3)
    
    # Add statistics text
    for i, (pos, data, label) in enumerate(zip(positions, data_to_plot, ['Classifiers', 'Regressors'])):
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        stats_text = f'{label}\nMean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
        y_pos = 1.0 if i == 0 else 0.95
        ax.text(pos, y_pos, stats_text, 
               ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(['Classifier-based\nNCMs', 'Regressor-based\nNCMs'], 
                       fontsize=12, fontweight='bold')
    ax.set_ylabel('Calibration Coverage', fontsize=12, fontweight='bold')
    ax.set_title('Classifier vs Regressor NCM Performance', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add arrow and text showing difference
    diff = means[0] - means[1]
    ax.annotate('', xy=(1, means[0]), xytext=(2, means[1]),
               arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
    ax.text(1.5, (means[0] + means[1]) / 2, f'+{diff:.3f}\n({diff*100:.1f}%)', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    return fig


def print_ncm_summary_stats(df):
    """
    Print summary statistics for NCM comparison.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ['Dataset Name', 'ncm', 'Empirical Coverage', 'split']
    """
    df_cal = df[df['Split'] == 'Calibration'].copy()
    
    print("=" * 70)
    print("NCM MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    # Overall stats
    stats = df_cal.groupby('ncm')['Empirical Coverage'].agg(['mean', 'std', 'min', 'max', 'count'])
    stats = stats.sort_values('mean', ascending=False)
    
    print("Calibration Coverage Statistics:")
    print(stats.to_string())
    print()
    
    # Classifier vs Regressor
    df_cal['ncm_type'] = df_cal['ncm'].apply(
        lambda x: 'Classifier' if x.startswith(('c', 'o')) else 'Regressor'
    )
    
    type_stats = df_cal.groupby('ncm_type')['Empirical Coverage'].agg(['mean', 'std', 'count'])
    print("\nBy NCM Type:")
    print(type_stats.to_string())
    print()
    
    # Statistical test
    from scipy import stats as sp_stats
    classifiers = df_cal[df_cal['ncm_type'] == 'Classifier']['Empirical Coverage']
    regressors = df_cal[df_cal['ncm_type'] == 'Regressor']['Empirical Coverage']
    
    t_stat, p_value = sp_stats.ttest_ind(classifiers, regressors)
    print(f"\nT-test (Classifier vs Regressor):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant? {'Yes' if p_value < 0.001 else 'No'} (α=0.001)")
    print()
    
    # Best/worst datasets
    print("\nMost Challenging Datasets (lowest mean coverage across NCMs):")
    dataset_means = df_cal.groupby('Dataset Name')['Empirical Coverage'].mean().sort_values()
    print(dataset_means.head(5).to_string())
    print()
    
    print("Easiest Datasets (highest mean coverage):")
    print(dataset_means.tail(5).to_string())
    print()
    
    print("=" * 70)


path_class = upstream["classification_summary_mapie"]
df = pd.read_excel(os.path.join(path_class["data"]))

# Create plots
fig1 = plot_ncm_coverage_comparison(df, output_path=os.path.join(product["data"],'ncm_comparison.png'))
fig2 = plot_ncm_heatmap(df, output_path=os.path.join(product["data"],'ncm_correlation_heatmap.png'))
fig3 = plot_classifier_vs_regressor_comparison(df, output_path=os.path.join(product["data"],'classifier_vs_regressor.png'))
print_ncm_summary_stats(df)

plt.show()