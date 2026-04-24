import pandas as pd
from pathlib import Path
import numpy as np
from tasks.assessment.utils import init_logging
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
from scipy import stats
from tasks.mapie_diagnostic import (
    plot_coverage_efficiency_classification,
    figure_spearman_classification,
    distance_by_adi_bins_classification,
    ADI_BIN_EDGES, ADI_BIN_LABELS
)


# + tags=["parameters"]
product = None
upstream = None
vega_models = None
mode = "classification"
data = ["BCF_MEYLAN"]
ncm = "crfecfp"
# -


#SCORE = "predicted0distance"
#SCORE_LABEL = f"Predicted {ncm} Distance"

#SCORE = "_probs_zero_distance"
#SCORE_LABEL = "Probs Zero Difference"

SCORE = "Set_Size"
SCORE_LABEL = "Predicted label set size"

df_models = pd.read_excel(vega_models, engine="openpyxl")
logger = init_logging(Path(product["nb"]).parent / "logs", "plots.log")


def coverage_by_adi_bins_classification(df, alpha=0.1, save_path=None):
    """
    Analyze coverage and prediction set size by ADI bins for classification.
    Focus on coverage rates (binary) rather than correlation.
    
    Args:
        df: DataFrame with columns ['ADI', 'In_Coverage', 'Set_Size']
        alpha: Significance level (e.g., 0.1 for 90% coverage)
        save_path: Path to save figure
    """
    # Bin ADI into groups
    if 'ADI_bin' not in df.columns:
        df['ADI_bin'] = pd.cut(df['ADI'], bins=ADI_BIN_EDGES, labels=ADI_BIN_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === PLOT 1: Coverage rate by ADI bin ===
    coverage_by_adi = df.groupby('ADI_bin')['In_Coverage'].agg(['mean', 'count', 'sem'])
    
    axes[0, 0].bar(range(len(coverage_by_adi)), coverage_by_adi['mean'], 
                   color='#2E7D32', alpha=0.7, edgecolor='black')
    axes[0, 0].errorbar(range(len(coverage_by_adi)), coverage_by_adi['mean'],
                        yerr=1.96*coverage_by_adi['sem'], fmt='none', 
                        color='black', capsize=5, label='95% CI')
    axes[0, 0].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Coverage Rate by Applicability Domain', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, 
                       label=f'Target {(1-alpha)*100:.0f}%')
    # axes[0, 0].set_xticks(range(len(coverage_by_adi)))
    # axes[0, 0].set_xticklabels(coverage_by_adi.index)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add sample sizes on bars
    for i, (idx, row) in enumerate(coverage_by_adi.iterrows()):
        axes[0, 0].text(i, row['mean'] + 0.05, f"n={row['count']}",
                       ha='center', fontsize=9, fontweight='bold')

    # === PLOT 2: Set size distribution by ADI bin ===
    set_size_data = [df[df['ADI_bin']==bin_label]['Set_Size'].values 
                     for bin_label in coverage_by_adi.index]
    bp = axes[0, 1].boxplot(set_size_data, labels=coverage_by_adi.index, 
                            patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4CAF50')
        patch.set_alpha(0.7)
    axes[0, 1].set_ylabel('Prediction Set Size', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Set Size Distribution by Applicability Domain', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # === PLOT 3: Singleton rate by ADI bin ===
    singleton_by_adi = df.groupby('ADI_bin').apply(
        lambda x: pd.Series({
            'singleton_rate': (x['Set_Size'] == 1).mean(),
            'count': len(x)
        })
    )
    
    axes[1, 0].bar(range(len(singleton_by_adi)), singleton_by_adi['singleton_rate'],
                   color='#FF9800', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Singleton Rate (Set Size = 1)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Certainty (Singleton Predictions) by Domain', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(range(len(singleton_by_adi)))
    axes[1, 0].set_xticklabels(singleton_by_adi.index, rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # === PLOT 4: Probability thresholds by ADI bin ===
    thresholds = df.groupby('ADI_bin').apply(
        lambda x: pd.Series({
            'P(size<1)': (x['Set_Size'] < 1).mean(),
            'P(size=1)': (x['Set_Size'] == 1).mean(),
            'P(size<=2)': (x['Set_Size'] <=2).mean(),
            'P(size>=3)': (x['Set_Size'] >=3).mean(),
        })
    )

    x = range(len(thresholds))

    for col in thresholds.columns:
        axes[1, 1].plot(x, thresholds[col], marker='o', label=col)

    axes[1, 1].set_ylabel('Probability of Set size', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Set Size Probabilities by ADI',
                        fontsize=14, fontweight='bold')

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(thresholds.index, rotation=0)

    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n=== Coverage by ADI Bin ===")
    print(coverage_by_adi[['mean', 'count']])
    print(f"\nOverall coverage: {df['In_Coverage'].mean():.1%}")
    print(f"Mean set size: {df['Set_Size'].mean():.3f}")
    print(f"Median set size: {df['Set_Size'].median():.1f}")
    print(f"Overall singleton rate: {(df['Set_Size'] == 1).mean():.1%}")
    
    # Statistical test: Chi-square for coverage differences across bins
    contingency = pd.crosstab(df['ADI_bin'], df['In_Coverage'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test for coverage differences: χ² = {chi2:.2f}, p = {p_val:.4f}")
    
    if p_val < 0.05:
        print("✓ Significant differences in coverage across ADI bins")
    else:
        print("✗ No significant differences in coverage across ADI bins")


def compare_datasets_coverage(combined_df, save_path=None):
    """
    Compare coverage rates across multiple datasets/endpoints.
    Better than Spearman for binary coverage outcomes.
    
    Args:
        combined_df: DataFrame with columns ['data', 'ADI', 'In_Coverage', 'Set_Size']
        save_path: Path to save figure
    """
    # Calculate metrics per dataset
    dataset_metrics = []
    for (name, split), g in combined_df.groupby(['data', 'split']):
        if len(g) < 10:
            continue
            
        metrics = {
            'data': name,
            'split': split,
            'n': len(g),
            'coverage': g['In_Coverage'].mean(),
            'mean_set_size': g['Set_Size'].mean(),
            'singleton_rate': (g['Set_Size'] == 1).mean(),
            'mean_adi': g['ADI'].mean(),
        }
        
        # Coverage by ADI quartiles
        if 'ADI' in g.columns:
            # --- Robust quartile binning ---
            try:
                adi_vals = g['ADI'].dropna()

                # Need at least 2 unique values to form bins
                if adi_vals.nunique() < 2:
                    raise ValueError("Not enough unique ADI values")

                tmp_bins = pd.qcut(adi_vals, q=4, duplicates='drop')

                if tmp_bins.cat.categories.size < 2:
                    raise ValueError("Too few bins after qcut")

                n_bins = tmp_bins.cat.categories.size
                labels = [f"Q{i+1}" for i in range(n_bins)]

                adi_bins = pd.qcut(adi_vals, q=n_bins, labels=labels, duplicates='drop')

                cov_by_quartile = g.loc[adi_vals.index].groupby(adi_bins)['In_Coverage'].mean()

                if len(cov_by_quartile) < 2:
                    raise ValueError("Empty or single-bin coverage")

                metrics['coverage_Q1'] = cov_by_quartile.iloc[0]
                metrics['coverage_Q4'] = cov_by_quartile.iloc[-1]
                metrics['coverage_diff_Q4_Q1'] = (
                    metrics['coverage_Q4'] - metrics['coverage_Q1']
                )

            except Exception:
                metrics['coverage_Q1'] = np.nan
                metrics['coverage_Q4'] = np.nan
                metrics['coverage_diff_Q4_Q1'] = np.nan
        
        dataset_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(dataset_metrics)\
                    .sort_values(['split','coverage'])    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === PLOT 1: Coverage rates by dataset and split ===

    # Create combined label: Dataset [Split]
    df_metrics['data_split'] = (
        df_metrics['data'].astype(str) + " [" +
        df_metrics['split'].astype(str) + "]"
    )

    # Sort within split then coverage
    df_plot = df_metrics.sort_values(['split', 'coverage'])

    colors_cov = [
        '#D32F2F' if cov < 0.85 else
        '#FFA726' if cov < 0.95 else
        '#2E7D32'
        for cov in df_plot['coverage']
    ]

    y_pos = np.arange(len(df_plot))

    axes[0, 0].barh(
        y_pos,
        df_plot['coverage'],
        color=colors_cov,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    axes[0, 0].axvline(x=0.9, color='red', linestyle='--',
                    linewidth=2, label='90% target')

    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(df_plot['data_split'], fontsize=9)
    axes[0, 0].set_xlabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Coverage by Dataset and Split (Sorted)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim(0, 1.05)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # === PLOT 2: Coverage vs mean ADI (split-aware, dynamic) ===

    import itertools

    marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', 'X'])
    splits = df_metrics['split'].dropna().unique()

    for sp, mk in zip(splits, marker_cycle):
        sub = df_metrics[df_metrics['split'] == sp]
        axes[0, 1].scatter(
            sub['mean_adi'],
            sub['coverage'],
            s=sub['n'] / 10,
            alpha=0.65,
            edgecolor='black',
            marker=mk,
            label=str(sp)
        )

    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Mean ADI', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Coverage vs Mean ADI by Split', fontsize=14, fontweight='bold')
    axes[0, 1].legend(title='Split')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add correlation
    valid_mask = df_metrics[['mean_adi', 'coverage']].notna().all(axis=1)
    if valid_mask.sum() > 2:
        rho, p = stats.spearmanr(df_metrics.loc[valid_mask, 'mean_adi'], 
                                 df_metrics.loc[valid_mask, 'Set_Size'])
        #rho, p = stats.kendalltau(df_metrics.loc[valid_mask, 'mean_adi'], 
        #                         df_metrics.loc[valid_mask, 'Set_Size'])        
        
        axes[0, 1].text(0.02, 0.98, f'ρ = {rho:.3f}\np = {p:.3f}',
                       transform=axes[0, 1].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === PLOT 3: ===
# === Panel 1,0: Coverage by ADI bins (split-aware) ===


    # Prepare data for plotting
    df_plot = []

    for sp, g in combined_df.groupby('split'):
        g['ADI_bin'] = pd.cut(g['ADI'], bins=ADI_BIN_EDGES, labels=ADI_BIN_LABELS)
        cov_by_bin = g.groupby('ADI_bin')['In_Coverage'].mean().reset_index()
        cov_by_bin['split'] = sp
        df_plot.append(cov_by_bin)

    df_plot = pd.concat(df_plot, ignore_index=True)

    # Plot bars
    splits = df_plot['split'].unique()
    x = np.arange(len(ADI_BIN_LABELS))  # positions for ADI bins
    width = 0.35  # width of each split's bar

    axes[1, 0].cla()  # clear previous panel

    for i, sp in enumerate(splits):
        sub = df_plot[df_plot['split'] == sp]
        sub = sub.set_index('ADI_bin').reindex(ADI_BIN_LABELS).reset_index()  # ensure order
        axes[1, 0].bar(
            x + i*width,
            sub['In_Coverage'],
            width=width,
            label=str(sp),
            alpha=0.7,
            edgecolor='black'
        )

    axes[1, 0].axhline(0.9, color='red', linestyle='--', linewidth=2, label='Target 90%')
    axes[1, 0].set_xticks(x + width*(len(splits)-1)/2)
    axes[1, 0].set_xticklabels(ADI_BIN_LABELS, rotation=0)
    axes[1, 0].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Coverage by ADI Range (Split-aware)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    axes[1, 0].legend(title='Split')

    
    # === PLOT 4: Singleton rate vs dataset size ===
    # Panel 1,1: Coverage vs dataset size (split-aware)
    axes[1, 1].clear()  # clear previous plot

    splits = df_metrics['split'].dropna().unique()
    marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', 'X'])

    for sp, mk in zip(splits, marker_cycle):
        sub = df_metrics[df_metrics['split'] == sp]
        axes[1, 1].scatter(
            sub['n'],
            sub['coverage'],
            s=100,
            alpha=0.7,
            edgecolor='black',
            marker=mk,
            label=str(sp)
        )

    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Dataset Size (log scale)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Coverage vs Dataset Size', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(title='Split')

    
    # Manual colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                               norm=plt.Normalize(vmin=0.8, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 1])
    cbar.set_label('Coverage', fontsize=10)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_metrics


def print_classification_summary(combined_df):
    """
    Print comprehensive summary for classification conformal prediction.
    """
    print("\n" + "="*70)
    print(" "*15 + "CLASSIFICATION CONFORMAL PREDICTION SUMMARY")
    print("="*70)
    
    # Overall metrics
    print(f"\n{'OVERALL PERFORMANCE':^70}")
    print("-"*70)
    print(f"Total Predictions:       {len(combined_df):,}")
    print(f"Coverage Rate:           {combined_df['In_Coverage'].mean():.1%}")
    print(f"Mean Set Size:           {combined_df['Set_Size'].mean():.3f}")
    print(f"Median Set Size:         {combined_df['Set_Size'].median():.0f}")
    print(f"Singleton Rate:          {(combined_df['Set_Size'] == 1).mean():.1%}")
    print(f"Max Set Size:            {combined_df['Set_Size'].max():.0f}")
    
    if (combined_df['Set_Size'] == 0).any():
        print(f"⚠️  Empty Sets:            {(combined_df['Set_Size'] == 0).sum()} ({(combined_df['Set_Size'] == 0).mean():.1%})")
    
    # By dataset
    print(f"\n{'PERFORMANCE BY DATASET':^70}")
    print("-"*70)
    
    dataset_summary = combined_df.groupby('data').agg({
        'In_Coverage': 'mean',
        'Set_Size': ['mean', 'median'],
        'ADI': 'mean'
    }).round(3)
    dataset_summary.columns = ['Coverage', 'Mean_Size', 'Median_Size', 'Mean_ADI']
    dataset_summary['Singleton%'] = (combined_df.groupby('data').apply(
        lambda x: (x['Set_Size'] == 1).mean()
    ) * 100).round(1)
    dataset_summary['n'] = combined_df.groupby('data').size()
    
    display(dataset_summary)
    
    # ADI analysis
    if 'ADI' in combined_df.columns:
        print(f"\n{'ADI STRATIFIED ANALYSIS':^70}")
        print("-"*70)
        
        adi_bins = pd.cut(combined_df['ADI'], bins=ADI_BIN_EDGES,
                         labels=ADI_BIN_LABELS)
        
        adi_analysis = combined_df.groupby(['split', adi_bins]).agg({
            'In_Coverage':['mean','count'],
            'Set_Size':'mean'
        }).round(3)
        adi_analysis.columns = ['Coverage', 'n', 'Mean_Size']
        adi_analysis['Singleton%'] = (combined_df.groupby(adi_bins).apply(
            lambda x: (x['Set_Size'] == 1).mean()
        ) * 100).round(1)
        
        display(adi_analysis)
        
        # Statistical test
        contingency = pd.crosstab(adi_bins, combined_df['In_Coverage'])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test: χ² = {chi2:.2f}, p = {p_val:.4f}")
        print(f"Result: {'Significant' if p_val < 0.05 else 'Not significant'} coverage differences across ADI bins")
    
        print(f"\n{'PERFORMANCE BY SPLIT':^70}")
        print("-"*70)

        split_summary = combined_df.groupby('split').agg({
            'In_Coverage':'mean',
            'Set_Size':['mean','median'],
            'ADI':'mean'
        }).round(3)

        split_summary.columns = ['Coverage','Mean_Size','Median_Size','Mean_ADI']
        split_summary['Singleton%'] = (
            combined_df.groupby('split')
            .apply(lambda x:(x['Set_Size']==1).mean()*100)
        ).round(1)

        display(split_summary)

    print("\n" + "="*70)


combined_rows = []
for key_star in upstream:
    for key in upstream[key_star]:
        # we only process the specified ncm here
        if key.split("_")[-1] != ncm:
            continue
        _data = key.replace("mapiecproba_", "").replace("mapiec_", "").replace(f"_{ncm}", "")
        #if _data in ["DEVTOX_PG"]:
        #    continue
        file_path = upstream[key_star][key]["data"]
        # Load classification results
        try:
            df_test = pd.read_excel(file_path, sheet_name="Prediction Intervals")
            if f'{_data}_true' in df_test.columns:
                df_test['correct'] = (df_test[f'{_data}_pred'] == df_test[f'{_data}_true'])

            df_test['data'] = _data
            df_test['split'] = 'Test'

            #if f'{_data}_probs_zero_distance' in df_test.columns:
            if f'{_data}{SCORE}' in df_test.columns:
                df_test = df_test.rename(columns={f'{_data}{SCORE}': SCORE})
            #combined_rows.append(df_test)
        except Exception:
            pass
        try:
            df_train = pd.read_excel(file_path, sheet_name="Training PI")
            if f'{_data}_true' in df_train.columns:
                df_train['correct'] = (df_train[f'{_data}_pred'] == df_train[f'{_data}_true'])

            if f'{_data}{SCORE}' in df_train.columns:
                df_train = df_train.rename(columns={f'{_data}{SCORE}': SCORE})                
            df_train['data'] = _data
            df_train['split'] = 'Training'        
            combined_rows.append(df_train)
        except Exception:
            pass

for _df in combined_rows:
    print(_df.columns)

combined_df = pd.concat(combined_rows, ignore_index=True)

# ========== SPEARMAN CORRELATION ANALYSIS (NO SPLITS) ==========
# Singleton indicator (1 if the prediction set is a singleton, 0 otherwise).
# This is the classification analog of interval width in regression:
#   - In regression: ADI correlates with interval width (point-wise, continuous)
#   - In classification: ADI correlates with is_singleton (point-wise, binary)
# Set_Size is misleading because LAC can produce empty sets, making the mean
# a mix of two effects. The singleton indicator isolates efficiency clearly:
# positive τ means higher ADI → more singleton (more certain) predictions.

combined_df['is_singleton'] = (combined_df['Set_Size'] == 1).astype(int)

_SCORE = 'is_singleton'
_SCORE_LABEL = 'Singleton prediction (Set Size = 1)'
_ADI_COL = "ADI"

# Global correlation (all data pooled)
rho, p = stats.kendalltau(
    combined_df[_ADI_COL],
    combined_df[_SCORE],
    variant="b"
)
print(f"\nGlobal Kendall τ-b (ADI vs {_SCORE_LABEL}) = {rho:.3f}, p = {p:.2e}")
print("Positive τ: higher ADI → more singleton predictions (better efficiency)")

# Per-dataset correlation analysis
rows = []
for name in combined_df['data'].unique():
    g = combined_df[combined_df['data'] == name].dropna(subset=[_ADI_COL, _SCORE])
    if len(g) > 10:
        tau, pval = stats.kendalltau(g[_ADI_COL], g[_SCORE], variant="b")
        singleton_rate = g['is_singleton'].mean() * 100
        rows.append({
            "data": name,
            "rho": tau,          # keep column name 'rho' for figure_spearman_classification
            "p": pval,
            "n": len(g),
            "singleton_rate": singleton_rate,   # used in panel B of figure
        })

corr_df = pd.DataFrame(rows).sort_values("rho")
display(corr_df)

print("\nInterpretation:")
print(f"Kendall τ-b measures monotonic association between ADI and singleton indicator.")
print(f"  Positive τ: Higher ADI → more singleton predictions (better efficiency in domain)")
print(f"  Negative τ: Higher ADI → fewer singletons (unexpected, check calibration)")
print(f"  τ ≈ 0: No monotonic relationship between ADI and prediction certainty")

# Save correlation results
corr_df.to_excel(product["data"], index=False)

# Generate Spearman figure (consistent with regression style; panel structure unchanged)
figure_spearman_classification(
    corr_df,
    score_col_label=_SCORE_LABEL,
    corr_label="rho",
    corr_title="Kendall τ-b (ADI vs Singleton)",
    save_path=product["plot"],
)

# Generate ADI bins analysis
distance_by_adi_bins_classification(
    combined_df,
    distance_col='Set_Size',
    distance_label=SCORE_LABEL,
    singleton_col='is_singleton',
    save_path=product["plot"].replace("spearman", "adi_bins_distance")
)

# ========== EXISTING VISUALIZATIONS  ==========

# Original coverage by ADI bins
HTML("<h3>Coverage and Uncertainty Stratified by Applicability Domain (ADI)</h3>") 
HTML("<p>Empirical coverage rate, prediction set size, and efficiency as a function of applicability domain index (ADI).</p>")


coverage_by_adi_bins_classification(
    combined_df, alpha=0.1,
    save_path=product["plot"].replace("spearman", "coverage_by_adi_bins"))

# Coverage efficiency plot (UNCHANGED)
print(combined_df[_SCORE].dtype)
print(combined_df[_SCORE].head())

dataset_stats = plot_coverage_efficiency_classification(
    combined_df,
    distance_col=SCORE,
    distance_label=SCORE_LABEL,
    save_path=product["plot"].replace("spearman", "coverage_efficiency"),
    max_labels_panel_a=50,
    annotate_top_n=4
)

dataset_stats.rename(columns={"mean_distance": f"mean_{_SCORE}",
                              "median_distance": f"median_{_SCORE}",
                              "std_distance": f"std_{_SCORE}"}
                              )
# Export
dataset_stats.to_excel(
    product["data"].replace("spearman", "coverage_efficiency_analysis"),
    index=False
)