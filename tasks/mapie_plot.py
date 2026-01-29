import pandas as pd
from pathlib import Path
import numpy as np
from tasks.assessment.utils import init_logging
from tasks.mapie_diagnostic import (
    plot_prediction_intervals,
    plot_interval_width_histogram,
    plot_prediction_intervals_index,
    mark_outlier,
    plot_coverage_efficiency_analysis
)
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, Markdown, HTML
from scipy.stats import spearmanr


# + tags=["parameters"]
product = None
upstream = None
vega_models = None
mode = "regression"
data = ["BCF_MEYLAN"]
ncm = "rfecfp"
# -


def figure_spearman(df, save_path=None):
    # Sort by correlation strength
    df_sorted = df.sort_values('rho').reset_index(drop=True)

    # Assign colors based on p-value significance
    colors = []
    for p in df_sorted['p']:
        if p < 0.001:
            colors.append('#2E7D32')  # Dark green - highly significant
        elif p < 0.05:
            colors.append('#FFA726')  # Orange - significant
        else:
            colors.append('#D32F2F')  # Red - not significant

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 14))

    # === PLOT 1: Correlation bars ===
    y_pos = np.arange(len(df_sorted))
    ax1.barh(y_pos, df_sorted['rho'], color=colors, alpha=0.8, 
            edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['data'], fontsize=9)
    ax1.set_xlabel('Spearman ρ', fontsize=12 )
    ax1.set_title('ADI vs Interval Width Correlation\n(Sorted by Strength)', 
                fontsize=12, pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.8, 0.2)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', label='p < 0.001'),
        Patch(facecolor='#FFA726', label='p < 0.05'),
        Patch(facecolor='#D32F2F', label='p ≥ 0.05')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # === PLOT 2: Scatter plot with dataset size ===
    # Color points by same significance scheme
    ax2.scatter(df_sorted['rho'], df_sorted['n'], c=colors, s=100, 
                alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Strength vs Dataset Size', 
                fontsize=12,  pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')  # Log scale for better visibility

    # Annotate a few interesting points (largest datasets or unusual cases)
    # Top 3 largest datasets
    top_n = df_sorted.nlargest(3, 'n')
    for _, row in top_n.iterrows():
        ax2.annotate(row['data'], 
                    xy=(row['rho'], row['n']), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Add legend
    ax2.legend(handles=legend_elements, loc='lower left', fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Mean ρ: {df['rho'].mean():.3f}")
    print(f"Median ρ: {df['rho'].median():.3f}")
    print(f"Range: [{df['rho'].min():.3f}, {df['rho'].max():.3f}]")
    print(f"\nSignificant correlations (p < 0.05): {(df['p'] < 0.05).sum()}/{len(df)}")
    print(f"Highly significant (p < 0.001): {(df['p'] < 0.001).sum()}/{len(df)}")
    print(f"\nNegative correlations: {(df['rho'] < 0).sum()}/{len(df)}")
    print(f"Positive correlations: {(df['rho'] > 0).sum()}/{len(df)}")
    print(f"\nDataset size range: {df['n'].min()} to {df['n'].max()}")
    print(f"Median dataset size: {df['n'].median():.0f}")

    # Check if dataset size correlates with correlation strength
    size_rho_corr = df['n'].corr(df['rho'], method='spearman')
    print(f"\nCorrelation between dataset size and ρ: {size_rho_corr:.3f}")
    print("(Does larger n lead to stronger/weaker correlations?)")
    
    
def coverage_by_adi_bins(df, save_path=None):
    # Bin ADI into groups
    df['ADI_bin'] = pd.cut(df['ADI'], bins=[0, 0.5, 0.75, 0.85, 1.0], 
                        labels=['Very Low\n(0-0.5)', 'Low\n(0.5-0.75)', 
                                'Moderate\n(0.75-0.85)', 'High\n(0.85-1.0)'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === PLOT 1: Coverage rate by ADI bin ===
    coverage_by_adi = df.groupby('ADI_bin')['covered'].agg(['mean', 'count'])
    coverage_by_adi['mean'].plot(kind='bar', ax=ax1, color='#2E7D32', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage Rate by Applicability Domain', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target 90%')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample sizes on bars
    for i, (idx, row) in enumerate(coverage_by_adi.iterrows()):
        ax1.text(i, row['mean'] + 0.02, f"n={row['count']}", 
                ha='center', fontsize=9, fontweight='bold')

    # === PLOT 2: Interval width by ADI bin ===
    df.boxplot(column='Relative Interval Width', by='ADI_bin', ax=ax2, patch_artist=True)
    ax2.set_ylabel('Relative Interval Width', fontsize=12, fontweight='bold')
    ax2.set_xlabel('ADI Range', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Interval Width by Applicability Domain', fontsize=14, fontweight='bold')
    ax2.get_figure().suptitle('')  # Remove auto-title from boxplot
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n=== Coverage by ADI Bin ===")
    print(coverage_by_adi)
    print(f"\nOverall coverage: {df['covered'].mean():.1%}")
    print(f"Mean interval width: {df['Relative Interval Width'].mean():.3f}")


df_models = pd.read_excel(vega_models, engine="openpyxl")
logger = init_logging(Path(product["nb"]).parent / "logs", "plots.log")

combined_rows = []
for key_star in upstream:
    logger.info(f"{key_star}")
    for key in upstream[key_star]:
        _data = key.replace("mapie_","").replace(f"_{ncm}","")
        if _data not in data:
            continue
        model_path = upstream[key_star][key].get("ncmodel", None)
        if model_path is None:
            continue
        sigma_model = {}
        with open(model_path, "rb") as f:
            sigma_model = pickle.load(f)
        logger.info(f"{ncm}\t{_data}")
        file_path = upstream[key_star][key]["data"]

        meta = pd.read_excel(file_path, sheet_name="Cover sheet", header=None, index_col=0).transpose()
        max = meta["Max"].iloc[0]
        min = meta["Min"].iloc[0]
        units = meta["Property Units"].iloc[0]
        name = meta["Property Name"].iloc[0]
        df_train = pd.read_excel(file_path, sheet_name="Training PI")
        df_train["Relative Interval Width"] = (df_train[f"{_data}_upper"] - df_train[f"{_data}_lower"])/(max-min)
        df_train['covered'] = (df_train[f"{_data}_true"] >= df_train[f"{_data}_lower"]) & (df_train[f'{_data}_true'] <= df_train[f"{_data}_upper"])

        df_test = pd.read_excel(file_path, sheet_name="Prediction Intervals")
        df_test["Relative Interval Width"] = (df_test[f"{_data}_upper"] - df_test[f"{_data}_lower"])/(max-min)
        df_test['covered'] = (df_test[f"{_data}_true"] >= df_test[f"{_data}_lower"]) & (df_test[f'{_data}_true'] <= df_test[f"{_data}_upper"])

        plot_interval_width_histogram(
            [df_train, df_test], model=_data, figsize=(12,3), bins="auto",
            labels=["Training", "Test"], show_residual_hist=True)
        plot_prediction_intervals(df_train, model=_data, title="Training")
        plot_prediction_intervals(df_test, model=_data, title="Test")



        if "ADI" in df_train.columns:
            # ---- TRAIN ROWS ----
            tmp_train = df_train[["ADI", "Relative Interval Width", f"{_data}_true", f"{_data}_pred", "covered"]].copy()
            tmp_train["data"] = _data
            tmp_train["split"] = "train"
            tmp_train["abs_residual"] = abs(
                tmp_train[f"{_data}_true"] - tmp_train[f"{_data}_pred"]
            )
            # ---- TEST ROWS ----
            tmp_test = df_test[["ADI", "Relative Interval Width", f"{_data}_true", f"{_data}_pred", "covered"]].copy()
            tmp_test["data"] = _data
            tmp_test["split"] = "test"
            tmp_test["abs_residual"] = abs(
                tmp_test[f"{_data}_true"] - tmp_test[f"{_data}_pred"]
            )
            coverage_by_adi_bins(pd.concat([tmp_train, tmp_test]), None)
            combined_rows.append(tmp_train)
            combined_rows.append(tmp_test)

if len(combined_rows)>0:
    combined_df = pd.concat(combined_rows, ignore_index=True)            
    combined_df["outlier"] = mark_outlier(combined_df, col='Relative Interval Width', low=0.05, up=0.95)
    combined_df = combined_df.loc[~combined_df["outlier"]]

    
    rho, p = spearmanr(
        combined_df["ADI"],
        combined_df["Relative Interval Width"]
    )
    #|ρ| < 0.2 → negligible
    #0.2–0.4 → weak
    # 0.4 → moderate+
    print(f"Global Spearman ρ = {rho:.3f}, p = {p:.2e}")

    plt.figure(figsize=(6,5))
    plt.hexbin(
        combined_df["ADI"],
        combined_df["Relative Interval Width"],
        gridsize=40,
        mincnt=1
    )
    plt.xlabel("ADI")
    plt.ylabel("Relative Interval Width")
    plt.title("All datasets pooled")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.show()


    rows = []
    for name, g in combined_df.groupby("data"):
        if len(g) > 10:
            r, p = spearmanr(g["ADI"], g["Relative Interval Width"])
            rows.append({
                "data": name,
                "rho": r,
                "p": p,
                "n": len(g)
            })
    corr_df = pd.DataFrame(rows).sort_values("rho")
    display(corr_df)


    # Spearman ρ is a rank-based monotonic association:
    # ρ > 0 → as ADI increases, RIW tends to increase
    # ρ < 0 → as ADI increases, RIW tends to decrease
    # ρ ≈ 0 → no monotonic relationship
    # Compounds with higher ADI tend to have narrower conformal intervals, and compounds with lower ADI tend to have wider intervals.
    # It does not assume linearity.
    # More “out-of-domain” points are sometimes getting smaller uncertainty.
    # Conformal interval width depends on: width ≈ quantile of |residuals|
    # ADI measures something like: distance from training manifold


    # Small multiples (selected datasets)
    for name in corr_df.sort_values("rho").head(4)["data"]:
        g = combined_df[combined_df["data"]==name]
        plt.figure(figsize=(3,3))
        plt.scatter(g["ADI"], g["Relative Interval Width"], s=8, alpha=0.4)
        plt.title(name)
        plt.xlabel("ADI")
        plt.ylabel("Relative Interval Width")
        plt.tight_layout()
        plt.show()

    corr_df.to_excel(product["data"])

    combined_df["abs_residual"].corr(
        combined_df["Relative Interval Width"],
        method="spearman"
    )

    figure_spearman(corr_df, product["plot"])    
    coverage_by_adi_bins(combined_df, product["plot"].replace("spearman", "coverage_analysis"))

    # NEW: Generate Figure 2
    dataset_stats = plot_coverage_efficiency_analysis(
        combined_df, 
        save_path=product["plot"].replace("spearman", "coverage_efficiency")
    )
    
    # Export statistics
    dataset_stats.to_excel(
        product["data"].replace(".xlsx", "_performance.xlsx"),
        index=False,
        #max_labels_panel_a=15,  # Fewer labels in Panel A
        #annotate_top_n=2        # Fewer annotations in Panel D        
    )    