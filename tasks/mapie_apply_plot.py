import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings
from tasks.assessment.utils import init_logging
from pathlib import Path


# + tags=["parameters"]
product = None
ncm_code = None
classification = False
vega_models = None
upstream = None
prefix = None
# -


# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
logger = init_logging(Path(product["nb"]).parent / "logs", "plots.log")

# -----------------------------
# CONFIGURATION - FIXED BINS
# -----------------------------

ADI_BINS = [0, 0.5, 0.75, 0.85, 1.0]
ADI_LABELS = ["Very Low", "Low", "Moderate", "High"]

CHUNK_SIZE = 200000  # Process data in chunks after loading (memory management)


# -----------------------------
# STREAMING STATISTICS CLASSES
# -----------------------------

@dataclass
class StreamingStats:
    """Welford's algorithm for online mean and variance"""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared differences from mean
    
    def update(self, value: float, weight: int = 1):
        """Update statistics with new value(s)"""
        for _ in range(weight):
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.M2 += delta * delta2
    
    def update_batch(self, values: np.ndarray):
        """Update with array of values"""
        for v in values:
            if not np.isnan(v):
                self.update(v)
    
    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


@dataclass
class TDigest:
    """Simplified t-digest for quantile estimation"""
    delta: float = 0.01  # Compression parameter
    centroids: List[tuple] = field(default_factory=list)  # (mean, weight)
    max_centroids: int = 100
    
    def add(self, value: float, weight: float = 1.0):
        """Add a value to the digest"""
        self.centroids.append((value, weight))
        if len(self.centroids) > self.max_centroids:
            self._compress()
    
    def add_batch(self, values: np.ndarray):
        """Add multiple values"""
        for v in values:
            if not np.isnan(v):
                self.add(v)
    
    def _compress(self):
        """Merge centroids to maintain compression"""
        if not self.centroids:
            return
        
        # Sort by value
        self.centroids.sort(key=lambda x: x[0])
        
        # Simple compression: merge adjacent centroids
        compressed = []
        current_mean, current_weight = self.centroids[0]
        
        for mean, weight in self.centroids[1:]:
            if len(compressed) < self.max_centroids // 2:
                # Merge with current
                total_weight = current_weight + weight
                current_mean = (current_mean * current_weight + mean * weight) / total_weight
                current_weight = total_weight
            else:
                compressed.append((current_mean, current_weight))
                current_mean, current_weight = mean, weight
        
        compressed.append((current_mean, current_weight))
        self.centroids = compressed
    
    def quantile(self, q: float) -> float:
        """Estimate quantile q in [0, 1]"""
        if not self.centroids:
            return np.nan
        
        self.centroids.sort(key=lambda x: x[0])
        total_weight = sum(w for _, w in self.centroids)
        target = q * total_weight
        
        cumsum = 0
        for mean, weight in self.centroids:
            cumsum += weight
            if cumsum >= target:
                return mean
        
        return self.centroids[-1][0]
    
    def histogram(self, bins: np.ndarray) -> np.ndarray:
        """Get histogram counts for given bins"""
        if not self.centroids:
            return np.zeros(len(bins) - 1)
        
        counts = np.zeros(len(bins) - 1)
        for mean, weight in self.centroids:
            bin_idx = np.digitize(mean, bins) - 1
            if 0 <= bin_idx < len(counts):
                counts[bin_idx] += weight
        
        return counts


@dataclass
class BinnedAggregator:
    """Aggregate statistics within bins"""
    bins: List[str] = field(default_factory=list)
    stats: Dict[str, StreamingStats] = field(default_factory=dict)
    digests: Dict[str, TDigest] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def __post_init__(self):
        for bin_label in self.bins:
            self.stats[bin_label] = StreamingStats()
            self.digests[bin_label] = TDigest()
    
    def update(self, bin_label: str, distance: float):
        """Update statistics for a bin"""
        if bin_label not in self.stats:
            return
        
        self.counts[bin_label] += 1
        self.stats[bin_label].update(distance)
        self.digests[bin_label].add(distance)
    
    def get_mean_distance(self) -> Dict[str, float]:
        return {k: v.mean if v.n > 0 else np.nan for k, v in self.stats.items()}
    
    def get_std_distance(self) -> Dict[str, float]:
        return {k: v.std if v.n > 0 else np.nan for k, v in self.stats.items()}
    
    def get_quantiles(self, quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[str, List[float]]:
        """Get quantiles for each bin"""
        return {k: [v.quantile(q) for q in quantiles] for k, v in self.digests.items()}
    
    def get_all_values_approximation(self, bin_label: str, n_samples: int = 1000) -> np.ndarray:
        """Approximate the distribution by sampling from centroids"""
        if bin_label not in self.digests or not self.digests[bin_label].centroids:
            return np.array([])
        
        centroids = self.digests[bin_label].centroids
        values = []
        weights = []
        for mean, weight in centroids:
            values.append(mean)
            weights.append(weight)
        
        # Sample proportionally to weights
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        samples = np.random.choice(values, size=min(n_samples, len(values)*10), p=weights)
        return samples


# -----------------------------
# MAIN AGGREGATOR
# -----------------------------

class ConformalAggregator:
    """Main aggregator for all models and global statistics"""
    
    def __init__(self):
        # Global aggregators
        self.global_adi = BinnedAggregator(bins=ADI_LABELS)
        self.global_all = StreamingStats()  # Overall statistics
        self.global_digest = TDigest()  # Overall distribution
        
        # Per-model aggregators
        self.model_adi: Dict[str, BinnedAggregator] = {}
        self.model_all: Dict[str, StreamingStats] = {}
        
        # Model metadata
        self.model_names = []
        self.total_chemicals = 0
    
    def process_dataframe(self, df: pd.DataFrame, model_name: str):
        """Process a dataframe chunk for a specific model"""
        
        # Initialize model aggregators if needed
        if model_name not in self.model_adi:
            self.model_adi[model_name] = BinnedAggregator(bins=ADI_LABELS)
            self.model_all[model_name] = StreamingStats()
            self.model_names.append(model_name)
        
        # Bin the data
        df['ADI_bin'] = pd.cut(df['ADI'], bins=ADI_BINS, labels=ADI_LABELS, include_lowest=True)
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=['ADI_bin', 'distance'])
        
        self.total_chemicals += len(df_clean)
        
        # Update global and per-model aggregators
        for _, row in df_clean.iterrows():
            adi_bin = row['ADI_bin']
            distance = row['distance']
            
            # Global updates
            self.global_adi.update(adi_bin, distance)
            self.global_all.update(distance)
            self.global_digest.add(distance)
            
            # Per-model updates
            self.model_adi[model_name].update(adi_bin, distance)
            self.model_all[model_name].update(distance)


# -----------------------------
# FILE PROCESSING
# -----------------------------

def process_files(upstream, prefix, ncm_code, max_files=300):
    """Process all upstream files with streaming"""
    
    aggregator = ConformalAggregator()
    
    files_processed = 0
    files_failed = 0
    
    print("Processing files...")
    
    for tag in upstream:
        for key in upstream[tag]:
            if files_processed >= max_files:
                break
            # Extract model name
            model_name = key.replace(prefix, "").replace(f"_{ncm_code}", "")
            filepath = upstream[tag][key]["results"]
            print(model_name)
            try:
                # Determine distance column name
                dist_col = f"{model_name}_predicted_distance"
                
                # Read entire Excel file (can't chunk Excel natively)
                df = pd.read_excel(
                    filepath,
                    sheet_name="Prediction Intervals",
                    usecols=["ADI", dist_col]
                )
                
                # Rename for consistency
                df = df.rename(columns={dist_col: 'distance'})
                
                # Process in chunks internally to avoid peak memory
                n_rows = len(df)
                for start_idx in range(0, n_rows, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n_rows)
                    chunk = df.iloc[start_idx:end_idx]
                    aggregator.process_dataframe(chunk, model_name)
                
                files_processed += 1
                if files_processed % 10 == 0:
                    print(f"  Processed {files_processed} files...")
                
            except Exception as e:
                files_failed += 1
                print(f"  ✗ Failed {model_name}: {str(e)}")
                continue
    
    print(f"\nProcessing complete: {files_processed} files processed, {files_failed} failed")
    print(f"Total predictions: {aggregator.total_chemicals:,}")
    
    return aggregator


# -----------------------------
# STATISTICS OUTPUT
# -----------------------------

def write_statistics(aggregator: ConformalAggregator, base_path: str):
    """Write comprehensive statistics to files"""
    
    # ===== 1. GLOBAL STATISTICS CSV =====
    global_stats_data = []
    
    mean_dist = aggregator.global_adi.get_mean_distance()
    std_dist = aggregator.global_adi.get_std_distance()
    quantiles = aggregator.global_adi.get_quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
    
    for label in ADI_LABELS:
        count = aggregator.global_adi.counts[label]
        mean = mean_dist[label]
        std = std_dist[label]
        q05, q25, median, q75, q95 = quantiles[label]
        
        global_stats_data.append({
            'ADI_Bin': label,
            'Count': count,
            'Percentage': 100 * count / aggregator.total_chemicals if aggregator.total_chemicals > 0 else 0,
            'Mean_Distance': mean,
            'Std_Distance': std,
            'Q05': q05,
            'Q25': q25,
            'Median': median,
            'Q75': q75,
            'Q95': q95
        })
    
    df_global = pd.DataFrame(global_stats_data)
    df_global.to_csv(f"{base_path}_global_stats.csv", index=False, float_format='%.4f')
    print(f"Global statistics saved to: {base_path}_global_stats.csv")
    
    # ===== 2. PER-MODEL STATISTICS CSV =====
    model_stats_data = []
    
    for model_name in aggregator.model_names:
        overall_mean = aggregator.model_all[model_name].mean
        overall_std = aggregator.model_all[model_name].std
        total_count = sum(aggregator.model_adi[model_name].counts.values())
        
        # Get stats by ADI bin for this model
        model_mean_dist = aggregator.model_adi[model_name].get_mean_distance()
        model_std_dist = aggregator.model_adi[model_name].get_std_distance()
        model_quantiles = aggregator.model_adi[model_name].get_quantiles([0.25, 0.5, 0.75])
        
        model_stats_data.append({
            'Model': model_name,
            'Total_Count': total_count,
            'Overall_Mean': overall_mean,
            'Overall_Std': overall_std,
            'VeryLow_Mean': model_mean_dist.get('Very Low', np.nan),
            'VeryLow_Median': model_quantiles.get('Very Low', [np.nan]*3)[1],
            'Low_Mean': model_mean_dist.get('Low', np.nan),
            'Low_Median': model_quantiles.get('Low', [np.nan]*3)[1],
            'Moderate_Mean': model_mean_dist.get('Moderate', np.nan),
            'Moderate_Median': model_quantiles.get('Moderate', [np.nan]*3)[1],
            'High_Mean': model_mean_dist.get('High', np.nan),
            'High_Median': model_quantiles.get('High', [np.nan]*3)[1],
        })
    
    df_models = pd.DataFrame(model_stats_data)
    df_models = df_models.sort_values('Overall_Mean')  # Sort by certainty (low = more certain)
    df_models.to_csv(f"{base_path}_model_stats.csv", index=False, float_format='%.4f')
    print(f"Model statistics saved to: {base_path}_model_stats.csv")
    
    # ===== 3. DETAILED MODEL-ADI MATRIX CSV =====
    model_adi_data = []
    
    for model_name in aggregator.model_names:
        model_mean_dist = aggregator.model_adi[model_name].get_mean_distance()
        model_counts = aggregator.model_adi[model_name].counts
        
        for label in ADI_LABELS:
            model_adi_data.append({
                'Model': model_name,
                'ADI_Bin': label,
                'Count': model_counts[label],
                'Mean_Distance': model_mean_dist[label]
            })
    
    df_model_adi = pd.DataFrame(model_adi_data)
    df_model_adi.to_csv(f"{base_path}_model_adi_matrix.csv", index=False, float_format='%.4f')
    print(f"Model-ADI matrix saved to: {base_path}_model_adi_matrix.csv")
    
    # ===== 4. SUMMARY TEXT REPORT =====
    with open(f"{base_path}_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONFORMAL PREDICTION ANALYSIS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Predictions: {aggregator.total_chemicals:,}\n")
        f.write(f"Number of Models: {len(aggregator.model_names)}\n")
        f.write(f"Overall Mean Distance: {aggregator.global_all.mean:.4f}\n")
        f.write(f"Overall Std Distance: {aggregator.global_all.std:.4f}\n\n")
        
        # Global statistics by ADI
        f.write("GLOBAL STATISTICS BY ADI BIN\n")
        f.write("-"*80 + "\n")
        f.write(f"{'ADI Bin':<15} {'Count':<12} {'%':<8} {'Mean':<10} {'Std':<10} {'Median':<10} {'Q25':<10} {'Q75':<10}\n")
        f.write("-"*80 + "\n")
        
        for label in ADI_LABELS:
            count = aggregator.global_adi.counts[label]
            pct = 100 * count / aggregator.total_chemicals if aggregator.total_chemicals > 0 else 0
            mean = mean_dist[label]
            std = std_dist[label]
            q25, median, q75 = quantiles[label][1], quantiles[label][2], quantiles[label][3]
            
            f.write(f"{label:<15} {count:<12,} {pct:<8.2f} {mean:<10.4f} {std:<10.4f} {median:<10.4f} {q25:<10.4f} {q75:<10.4f}\n")
        
        f.write("\n\n")
        
        # Model rankings
        model_stats = [(name, aggregator.model_all[name].mean, sum(aggregator.model_adi[name].counts.values())) 
                       for name in aggregator.model_names]
        model_stats.sort(key=lambda x: x[1])
        
        f.write("MODEL RANKINGS (BY MEAN DISTANCE - LOWER = MORE CERTAIN)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Model':<50} {'Mean Distance':<15} {'N Predictions':<15}\n")
        f.write("-"*80 + "\n")
        
        for i, (name, mean, count) in enumerate(model_stats, 1):
            f.write(f"{i:<6} {name[:50]:<50} {mean:<15.4f} {count:<15,}\n")
        
        f.write("\n\n")
        
        # Top 10 and Bottom 10
        f.write("TOP 10 MOST CERTAIN MODELS (Lowest Mean Distance)\n")
        f.write("-"*80 + "\n")
        for i, (name, mean, count) in enumerate(model_stats[:10], 1):
            f.write(f"{i:2d}. {name[:60]:<60} {mean:.4f}\n")
        
        f.write("\n")
        
        f.write("TOP 10 MOST UNCERTAIN MODELS (Highest Mean Distance)\n")
        f.write("-"*80 + "\n")
        for i, (name, mean, count) in enumerate(reversed(model_stats[-10:]), 1):
            f.write(f"{i:2d}. {name[:60]:<60} {mean:.4f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {base_path}_summary.txt")


# -----------------------------
# PLOTTING FUNCTIONS
# -----------------------------

def plot_global_analysis(aggregator: ConformalAggregator, save_path: str):
    """Create comprehensive global analysis plots"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ===== Row 1: Summary Statistics =====
    
    # A: Mean distance vs ADI (with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    mean_dist = aggregator.global_adi.get_mean_distance()
    std_dist = aggregator.global_adi.get_std_distance()
    
    x_pos = np.arange(len(ADI_LABELS))
    means = [mean_dist[k] for k in ADI_LABELS]
    stds = [std_dist[k] for k in ADI_LABELS]
    
    ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax1.set_ylabel('Mean Predicted Distance', fontsize=11, fontweight='bold')
    ax1.set_title('A: Uncertainty vs Applicability Domain', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # B: Sample counts per ADI bin
    ax2 = fig.add_subplot(gs[0, 1])
    counts = [aggregator.global_adi.counts[k] for k in ADI_LABELS]
    bars = ax2.bar(x_pos, counts, alpha=0.7, color='coral', edgecolor='darkred')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax2.set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
    ax2.set_title('B: Sample Distribution by ADI', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # C: Box plot of distances by ADI
    ax3 = fig.add_subplot(gs[0, 2])
    quantiles = aggregator.global_adi.get_quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
    
    box_data = []
    for label in ADI_LABELS:
        q = quantiles[label]
        if not any(np.isnan(q)):
            box_data.append({
                'label': label,
                'whislo': q[0],  # 5th percentile
                'q1': q[1],      # 25th percentile
                'med': q[2],     # 50th percentile (median)
                'q3': q[3],      # 75th percentile
                'whishi': q[4],  # 95th percentile
            })
    
    if box_data:
        bp = ax3.bxp(box_data, positions=range(len(box_data)), 
                     widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
    
    ax3.set_xticks(range(len(ADI_LABELS)))
    ax3.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax3.set_ylabel('Predicted Distance', fontsize=11, fontweight='bold')
    ax3.set_title('C: Distance Distribution (Quantiles)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ===== Row 2: Distributions =====
    
    # D: Violin plots by ADI (using t-digest approximations)
    ax4 = fig.add_subplot(gs[1, :2])
    
    positions = []
    violin_data = []
    for i, label in enumerate(ADI_LABELS):
        samples = aggregator.global_adi.get_all_values_approximation(label, n_samples=1000)
        if len(samples) > 0:
            positions.append(i)
            violin_data.append(samples)
    
    if violin_data:
        parts = ax4.violinplot(violin_data, positions=positions, widths=0.7,
                               showmeans=True, showmedians=True)
        
        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
            pc.set_edgecolor('navy')
    
    ax4.set_xticks(range(len(ADI_LABELS)))
    ax4.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax4.set_ylabel('Predicted Distance', fontsize=11, fontweight='bold')
    ax4.set_title('D: Distance Distribution by ADI (Violin Plot)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # E: Overall distance histogram
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Get approximate histogram from global t-digest
    hist_bins = np.linspace(0, aggregator.global_all.mean + 3*aggregator.global_all.std, 50)
    hist_counts = aggregator.global_digest.histogram(hist_bins)
    
    # Plot as histogram
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    ax5.bar(bin_centers, hist_counts, width=np.diff(hist_bins), 
            alpha=0.7, color='purple', edgecolor='darkviolet')
    ax5.set_xlabel('Predicted Distance', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('E: Overall Distance Distribution', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ===== Row 3: Detailed Statistics =====
    
    # F: Cumulative distribution by ADI
    ax6 = fig.add_subplot(gs[2, :2])
    
    for label in ADI_LABELS:
        samples = aggregator.global_adi.get_all_values_approximation(label, n_samples=1000)
        if len(samples) > 0:
            sorted_samples = np.sort(samples)
            cumulative = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
            ax6.plot(sorted_samples, cumulative, label=label, linewidth=2, alpha=0.8)
    
    ax6.set_xlabel('Predicted Distance', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax6.set_title('F: Cumulative Distribution Functions by ADI', fontsize=12, fontweight='bold')
    ax6.legend(title='ADI Bin', loc='lower right')
    ax6.grid(alpha=0.3, linestyle='--')
    
    # G: Summary statistics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Create summary table
    table_data = [['ADI Bin', 'N', 'Mean', 'Std', 'Median']]
    for label in ADI_LABELS:
        count = aggregator.global_adi.counts[label]
        mean = aggregator.global_adi.get_mean_distance()[label]
        std = aggregator.global_adi.get_std_distance()[label]
        median = aggregator.global_adi.get_quantiles([0.5])[label][0]
        
        table_data.append([
            label,
            f'{count:,}',
            f'{mean:.3f}' if not np.isnan(mean) else 'N/A',
            f'{std:.3f}' if not np.isnan(std) else 'N/A',
            f'{median:.3f}' if not np.isnan(median) else 'N/A'
        ])
    
    table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax7.set_title('G: Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(f'Conformal Prediction Analysis - Global Summary\n({aggregator.total_chemicals:,} predictions across {len(aggregator.model_names)} models)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Global analysis saved to: {save_path}")
    plt.close()


def plot_model_comparison(aggregator: ConformalAggregator, save_path: str, top_n: int = 10):
    """Compare performance across models"""
    
    # Calculate overall mean distance per model
    model_stats = []
    for model_name in aggregator.model_names:
        mean_distances = aggregator.model_adi[model_name].get_mean_distance()
        overall_mean = aggregator.model_all[model_name].mean
        overall_std = aggregator.model_all[model_name].std
        total_count = sum(aggregator.model_adi[model_name].counts.values())
        model_stats.append((model_name, overall_mean, overall_std, total_count))
    
    # Sort by mean distance
    model_stats.sort(key=lambda x: x[1])
    
    # Select top and bottom models
    n_models = len(model_stats)
    if n_models > top_n * 2:
        selected = model_stats[:top_n] + model_stats[-top_n:]
        title_suffix = f"(Top & Bottom {top_n})"
    else:
        selected = model_stats
        title_suffix = "(All Models)"
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Model ranking by mean distance
    ax1 = fig.add_subplot(gs[0, :])
    models = [s[0] for s in selected]
    means = [s[1] for s in selected]
    stds = [s[2] for s in selected]
    
    y_pos = np.arange(len(models))
    colors = ['green' if i < top_n else 'red' for i in range(len(models))]
    
    ax1.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.6, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlabel('Mean Predicted Distance', fontsize=11, fontweight='bold')
    ax1.set_title(f'Model Ranking by Uncertainty {title_suffix}', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # Add mean value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(mean + std + 0.1, i, f'{mean:.2f}', va='center', fontsize=8)
    
    # Plot 2: Distance vs ADI for top models
    ax2 = fig.add_subplot(gs[1, 0])
    for model_name in models[:5]:  # Top 5 most certain
        mean_dist = aggregator.model_adi[model_name].get_mean_distance()
        distances = [mean_dist[k] for k in ADI_LABELS]
        ax2.plot(ADI_LABELS, distances, marker='o', label=model_name[:20], linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('ADI Bin', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Predicted Distance', fontsize=11, fontweight='bold')
    ax2.set_title('Top 5 Models: Distance vs ADI', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    
    # Plot 3: Distance vs ADI for bottom models
    ax3 = fig.add_subplot(gs[1, 1])
    for model_name in models[-5:]:  # Bottom 5 most uncertain
        mean_dist = aggregator.model_adi[model_name].get_mean_distance()
        distances = [mean_dist[k] for k in ADI_LABELS]
        ax3.plot(ADI_LABELS, distances, marker='o', label=model_name[:20], linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('ADI Bin', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Predicted Distance', fontsize=11, fontweight='bold')
    ax3.set_title('Bottom 5 Models: Distance vs ADI', fontsize=12, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    
    # Plot 4: Violin plot comparison for selected models
    ax4 = fig.add_subplot(gs[2, :])
    
    # Show violin plots for top 3 models across ADI bins
    top_3_models = models[:3]
    positions_base = np.arange(len(ADI_LABELS)) * 4
    
    for i, model_name in enumerate(top_3_models):
        for j, label in enumerate(ADI_LABELS):
            samples = aggregator.model_adi[model_name].get_all_values_approximation(label, n_samples=500)
            if len(samples) > 10:
                pos = positions_base[j] + i
                parts = ax4.violinplot([samples], positions=[pos], widths=0.8,
                                      showmeans=True, showmedians=False)
                # Color code by model
                color = plt.cm.Set3(i)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
    
    # Set up x-axis
    ax4.set_xticks(positions_base + 1)
    ax4.set_xticklabels(ADI_LABELS)
    ax4.set_xlabel('ADI Bin', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted Distance', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution Comparison: Top 3 Models by ADI', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.Set3(i), alpha=0.7, label=top_3_models[i][:20]) 
                      for i in range(len(top_3_models))]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    fig.suptitle(f'Model Comparison Analysis\n({len(aggregator.model_names)} models total)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to: {save_path}")
    plt.close()


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__" or True:  # Works in both script and notebook mode
    
    print("="*60)
    print("CONFORMAL PREDICTION ANALYSIS - STREAMING MODE")
    print("="*60)
    print()
    
    # Process all files
    aggregator = process_files(upstream, prefix, ncm_code, max_files=2)
    
    # Generate plots
    base_path = product["results"].rsplit('.', 1)[0]
    
    print("\nGenerating outputs...")
    write_statistics(aggregator, base_path)
    plot_global_analysis(aggregator, f"{base_path}_global.png")
    plot_model_comparison(aggregator, f"{base_path}_models.png", top_n=10)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nOutput files:")
    print(f"  - {base_path}_summary.txt")
    print(f"  - {base_path}_global_stats.csv")
    print(f"  - {base_path}_model_stats.csv")
    print(f"  - {base_path}_model_adi_matrix.csv")
    print(f"  - {base_path}_global.png")
    print(f"  - {base_path}_models.png")
    print("="*60)