import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import warnings

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


# + tags=["parameters"]
product = None
ncm_code = None
classification = False
vega_models = None
upstream = None
prefix = None
# -


# -----------------------------
# CONFIGURATION - FIXED BINS
# -----------------------------

ADI_BINS = [0, 0.5, 0.75, 0.85, 1.0]
ADI_LABELS = ["Very Low", "Low", "Moderate", "High"]

DIST_BINS = [0, 0.5, 1.5, 2.5, 3.5, np.inf]
DIST_LABELS = ["0", "1", "2", "3", "4+"]

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
    
    def get_quantiles(self, q: List[float]) -> Dict[str, List[float]]:
        """Get quantiles for each bin"""
        return {k: [v.quantile(qi) for qi in q] for k, v in self.digests.items()}


# -----------------------------
# MAIN AGGREGATOR
# -----------------------------

class ConformalAggregator:
    """Main aggregator for all models and global statistics"""
    
    def __init__(self):
        # Global aggregators
        self.global_adi = BinnedAggregator(bins=ADI_LABELS)
        self.global_dist = BinnedAggregator(bins=DIST_LABELS)
        
        # Per-model aggregators
        self.model_adi: Dict[str, BinnedAggregator] = {}
        self.model_dist: Dict[str, BinnedAggregator] = {}
        
        # Joint ADI × Distance histogram (streaming counts)
        self.joint_histogram = defaultdict(lambda: defaultdict(int))
        
        # Model metadata
        self.model_names = []
        self.total_chemicals = 0
    
    def process_dataframe(self, df: pd.DataFrame, model_name: str):
        """Process a dataframe chunk for a specific model"""
        
        # Initialize model aggregators if needed
        if model_name not in self.model_adi:
            self.model_adi[model_name] = BinnedAggregator(bins=ADI_LABELS)
            self.model_dist[model_name] = BinnedAggregator(bins=DIST_LABELS)
            self.model_names.append(model_name)
        
        # Bin the data
        df['ADI_bin'] = pd.cut(df['ADI'], bins=ADI_BINS, labels=ADI_LABELS, include_lowest=True)
        df['D_bin'] = pd.cut(df['distance'], bins=DIST_BINS, labels=DIST_LABELS, include_lowest=True)
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=['ADI_bin', 'D_bin', 'distance'])
        
        self.total_chemicals += len(df_clean)
        
        # Update global and per-model aggregators
        for _, row in df_clean.iterrows():
            adi_bin = row['ADI_bin']
            dist_bin = row['D_bin']
            distance = row['distance']
            
            # Global updates
            self.global_adi.update(adi_bin, distance)
            self.global_dist.update(dist_bin, distance)
            
            # Per-model updates
            self.model_adi[model_name].update(adi_bin, distance)
            self.model_dist[model_name].update(dist_bin, distance)
            
            # Joint histogram
            self.joint_histogram[adi_bin][dist_bin] += 1
    
    def get_joint_histogram_normalized(self) -> np.ndarray:
        """Get normalized joint ADI × Distance histogram"""
        hist_matrix = np.array([
            [self.joint_histogram[adi][dist] for dist in DIST_LABELS]
            for adi in ADI_LABELS
        ], dtype=float)
        
        # Normalize each row
        row_sums = hist_matrix.sum(axis=1, keepdims=True)
        hist_matrix = np.divide(
            hist_matrix, row_sums,
            out=np.zeros_like(hist_matrix),
            where=row_sums != 0
        )
        
        return hist_matrix


# -----------------------------
# FILE PROCESSING
# -----------------------------

def process_files(upstream, prefix, ncm_code, max=3):
    """Process all upstream files with streaming"""
    
    aggregator = ConformalAggregator()
    
    files_processed = 0
    files_failed = 0
    
    for tag in upstream:
        for key in upstream[tag]:
            if files_processed > max:
                break
            # Extract model name
            model_name = key.replace(prefix, "").replace(f"_{ncm_code}", "")
            filepath = upstream[tag][key]["results"]
            
            print(f"Processing {model_name}...")
            
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
                print(f"  ✓ Processed {model_name} ({n_rows:,} rows)")
                
            except Exception as e:
                files_failed += 1
                print(f"  ✗ Failed {model_name}: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Files processed: {files_processed}")
    print(f"  Files failed: {files_failed}")
    print(f"  Total chemicals: {aggregator.total_chemicals:,}")
    print(f"  Models: {len(aggregator.model_names)}")
    print(f"{'='*60}\n")
    
    return aggregator


# -----------------------------
# PLOTTING FUNCTIONS
# -----------------------------

def plot_global_analysis(aggregator: ConformalAggregator, save_path: str):
    """Create comprehensive global analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # A: Mean distance vs ADI (with error bars)
    mean_dist = aggregator.global_adi.get_mean_distance()
    std_dist = aggregator.global_adi.get_std_distance()
    
    x_pos = np.arange(len(ADI_LABELS))
    means = [mean_dist[k] for k in ADI_LABELS]
    stds = [std_dist[k] for k in ADI_LABELS]
    
    axes[0, 0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Predicted Distance', fontsize=11)
    axes[0, 0].set_title('A: Uncertainty vs Applicability Domain', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # B: Sample counts per ADI bin
    counts = [aggregator.global_adi.counts[k] for k in ADI_LABELS]
    axes[0, 1].bar(x_pos, counts, alpha=0.7, color='coral')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Number of Predictions', fontsize=11)
    axes[0, 1].set_title('B: Sample Distribution by ADI', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # C: Distance distribution by ADI (stacked bar)
    hist_matrix = aggregator.get_joint_histogram_normalized()
    
    bottom = np.zeros(len(ADI_LABELS))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(DIST_LABELS)))
    
    for i, (dlab, color) in enumerate(zip(DIST_LABELS, colors)):
        axes[1, 0].bar(
            x_pos,
            hist_matrix[:, i],
            bottom=bottom,
            label=dlab,
            color=color,
            alpha=0.8
        )
        bottom += hist_matrix[:, i]
    
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Fraction of Predictions', fontsize=11)
    axes[1, 0].set_title('C: Distance Distribution by ADI', fontsize=12, fontweight='bold')
    axes[1, 0].legend(title='Predicted\nDistance', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].set_ylim([0, 1])
    
    # D: Mean distance by distance bin (sanity check)
    mean_by_dist = aggregator.global_dist.get_mean_distance()
    counts_by_dist = [aggregator.global_dist.counts[k] for k in DIST_LABELS]
    
    x_pos_dist = np.arange(len(DIST_LABELS))
    means_dist = [mean_by_dist[k] for k in DIST_LABELS]
    
    ax_d = axes[1, 1]
    ax_d_twin = ax_d.twinx()
    
    ax_d.plot(x_pos_dist, means_dist, marker='o', linewidth=2, color='darkgreen', label='Mean Distance')
    ax_d_twin.bar(x_pos_dist, counts_by_dist, alpha=0.3, color='gray', label='Count')
    
    ax_d.set_xticks(x_pos_dist)
    ax_d.set_xticklabels(DIST_LABELS)
    ax_d.set_ylabel('Mean Distance', fontsize=11, color='darkgreen')
    ax_d_twin.set_ylabel('Count', fontsize=11, color='gray')
    ax_d.set_title('D: Distance Bins Validation', fontsize=12, fontweight='bold')
    ax_d.tick_params(axis='y', labelcolor='darkgreen')
    ax_d_twin.tick_params(axis='y', labelcolor='gray')
    ax_d.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Global analysis saved to: {save_path}")
    plt.close()


def plot_model_comparison(aggregator: ConformalAggregator, save_path: str, top_n: int = 10):
    """Compare performance across models"""
    
    # Calculate overall mean distance per model
    model_stats = []
    for model_name in aggregator.model_names:
        mean_distances = aggregator.model_adi[model_name].get_mean_distance()
        overall_mean = np.nanmean([mean_distances[k] for k in ADI_LABELS])
        total_count = sum(aggregator.model_adi[model_name].counts.values())
        model_stats.append((model_name, overall_mean, total_count))
    
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
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Model ranking by mean distance
    models = [s[0] for s in selected]
    means = [s[1] for s in selected]
    counts = [s[2] for s in selected]
    
    y_pos = np.arange(len(models))
    colors = ['green' if i < top_n else 'red' for i in range(len(models))]
    
    axes[0].barh(y_pos, means, color=colors, alpha=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models, fontsize=8)
    axes[0].set_xlabel('Mean Predicted Distance', fontsize=11)
    axes[0].set_title(f'Model Ranking by Uncertainty {title_suffix}', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot 2: Distance vs ADI for selected models
    for model_name in models[:5]:  # Show top 5 in detail
        mean_dist = aggregator.model_adi[model_name].get_mean_distance()
        distances = [mean_dist[k] for k in ADI_LABELS]
        axes[1].plot(ADI_LABELS, distances, marker='o', label=model_name, linewidth=2, alpha=0.7)
    
    axes[1].set_xlabel('ADI Bin', fontsize=11)
    axes[1].set_ylabel('Mean Predicted Distance', fontsize=11)
    axes[1].set_title('Distance vs ADI for Top 5 Models', fontsize=12, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to: {save_path}")
    plt.close()


def plot_heatmap(aggregator: ConformalAggregator, save_path: str):
    """Create heatmap of ADI × Distance joint distribution"""
    
    hist_matrix = aggregator.get_joint_histogram_normalized()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(hist_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(DIST_LABELS)))
    ax.set_yticks(np.arange(len(ADI_LABELS)))
    ax.set_xticklabels(DIST_LABELS)
    ax.set_yticklabels(ADI_LABELS)
    
    # Labels
    ax.set_xlabel('Predicted Distance Bin', fontsize=12)
    ax.set_ylabel('ADI Bin', fontsize=12)
    ax.set_title('Joint Distribution: ADI × Distance', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(ADI_LABELS)):
        for j in range(len(DIST_LABELS)):
            text = ax.text(j, i, f'{hist_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")
    plt.close()


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__" or True:  # Works in both script and notebook mode
    
    print("="*60)
    print("CONFORMAL PREDICTION ANALYSIS - STREAMING MODE")
    print("="*60)
    
    # Process all files
    aggregator = process_files(upstream, prefix, ncm_code, max=3)
    
    # Generate plots
    base_path = product["results"].rsplit('.', 1)[0]
    
    plot_global_analysis(aggregator, f"{base_path}_global.png")
    plot_model_comparison(aggregator, f"{base_path}_models.png", top_n=10)
    plot_heatmap(aggregator, f"{base_path}_heatmap.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nGlobal - Mean Distance by ADI:")
    mean_dist = aggregator.global_adi.get_mean_distance()
    for label in ADI_LABELS:
        count = aggregator.global_adi.counts[label]
        print(f"  {label:12s}: {mean_dist[label]:.3f} ± {aggregator.global_adi.get_std_distance()[label]:.3f} (n={count:,})")
    
    print("\nGlobal - Distribution across Distance Bins:")
    for label in DIST_LABELS:
        count = aggregator.global_dist.counts[label]
        pct = 100 * count / aggregator.total_chemicals if aggregator.total_chemicals > 0 else 0
        print(f"  {label:12s}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)