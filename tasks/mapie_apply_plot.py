import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy.stats import spearmanr
import warnings
import pickle
from pathlib import Path
from tasks.assessment.utils import init_logging
from tasks.interval_scaler import IntervalScaler


# + tags=["parameters"]
product = None
ncm_code = None
classification = None  # True for classification, False for regression
vega_models = None
upstream = None
prefix = None
max_files = None
data = None
# -

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
logger = init_logging(Path(product["nb"]).parent / "logs", "plots.log")

# -----------------------------
# CONFIGURATION - FIXED BINS
# -----------------------------

ADI_BINS = [0, 0.5, 0.75, 0.85, 1.0]
ADI_LABELS = ["Very Low", "Low", "Moderate", "High"]

CHUNK_SIZE = 50000  # Process data in chunks after loading (memory management)

# Determine metric name based on task type
METRIC_NAME = "predicted_distance" if classification else "Interval_Width"
METRIC_LABEL = "Predicted Distance" if classification else "Relative Interval Width"
METRIC_COL_SUFFIX = "predicted_distance" if classification else "Interval_Width"


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
    
    def merge(self, other: "StreamingStats"):
        """Merge another StreamingStats into this one"""
        if other.n == 0:
            return

        total_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean = (self.n*self.mean + other.n*other.mean)/total_n
        self.M2 += other.M2 + delta**2 * self.n*other.n/total_n
        self.n = total_n

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
    
    def update(self, bin_label: str, metric_value: float):
        """Update statistics for a bin"""
        if bin_label not in self.stats:
            return
        
        self.counts[bin_label] += 1
        self.stats[bin_label].update(metric_value)
        self.digests[bin_label].add(metric_value)
    
    def get_mean_metric(self) -> Dict[str, float]:
        return {k: v.mean if v.n > 0 else np.nan for k, v in self.stats.items()}
    
    def get_std_metric(self) -> Dict[str, float]:
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
    
    def __init__(self, is_classification: bool = True):
        # Task type
        self.is_classification = is_classification
        
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
        df_clean = df.dropna(subset=['ADI_bin', 'metric'])
        
        self.total_chemicals += len(df_clean)
        
        # Update global and per-model aggregators
        for _, row in df_clean.iterrows():
            adi_bin = row['ADI_bin']
            metric_value = row['metric']
            
            # Global updates
            self.global_adi.update(adi_bin, metric_value)
            self.global_all.update(metric_value)
            self.global_digest.add(metric_value)
            
            # Per-model updates
            self.model_adi[model_name].update(adi_bin, metric_value)
            self.model_all[model_name].update(metric_value)
    
    def save(self, filepath: str):
        """Save aggregator state to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Aggregator saved to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ConformalAggregator':
        """Load aggregator state from disk"""
        with open(filepath, 'rb') as f:
            aggregator = pickle.load(f)
        
        # Backward compatibility: add is_classification attribute if missing
        if not hasattr(aggregator, 'is_classification'):
            logger.warning("Loaded old aggregator format without is_classification attribute")
            logger.warning("Defaulting to is_classification=True (classification mode)")
            aggregator.is_classification = True
        
        return aggregator
    
    def filter_models(self, excluded_models: List[str]) -> "ConformalAggregator":
        """
        Return a new ConformalAggregator containing only models not in excluded_models.
        Global statistics are recomputed exactly from the retained models.

        Parameters
        ----------
        excluded_models : List[str]
            Model names to exclude.

        Returns
        -------
        ConformalAggregator
            Filtered aggregator with updated global stats.
        """
        excluded_models = set(excluded_models)
        new_aggr = ConformalAggregator(is_classification=self.is_classification)

        for model_name in self.model_names:
            if model_name in excluded_models:
                continue

            # Copy model-level stats
            new_aggr.model_names.append(model_name)
            new_aggr.model_adi[model_name] = self.model_adi[model_name]
            new_aggr.model_all[model_name] = self.model_all[model_name]

            # Merge into global ADI bins
            for bin_label in ADI_LABELS:
                new_aggr.global_adi.counts[bin_label] += self.model_adi[model_name].counts[bin_label]

                # Merge StreamingStats (means & variance)
                new_aggr.global_adi.stats[bin_label].merge(self.model_adi[model_name].stats[bin_label])

                # Merge TDigest centroids
                new_aggr.global_adi.digests[bin_label].centroids.extend(
                    self.model_adi[model_name].digests[bin_label].centroids
                )

            # Merge overall stats
            new_aggr.global_all.merge(self.model_all[model_name])

            # Merge global t-digest
            if hasattr(self.global_digest, "centroids"):
                new_aggr.global_digest.centroids.extend(
                    self.global_digest.centroids
                )

            # Update total count
            new_aggr.total_chemicals += sum(self.model_adi[model_name].counts.values())

        return new_aggr



# -----------------------------
# FILE PROCESSING WITH CACHING
# -----------------------------

def process_files(upstream, prefix, ncm_code, is_classification, max_files=300,
                  cache_path=None, scaler=None, datasets=None):
    """Process all upstream files with streaming and caching support"""
    
    # Check if cached aggregator exists
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached aggregator from: {cache_path}")
        aggregator = ConformalAggregator.load(cache_path)
        logger.info(f"Loaded: {aggregator.total_chemicals:,} predictions across {len(aggregator.model_names)} models")
        return aggregator
    
    logger.info("No cache found, processing files...")
    aggregator = ConformalAggregator(is_classification)
    
    files_processed = 0
    files_failed = 0
    
    # Determine column suffix based on task type
    col_suffix = METRIC_COL_SUFFIX
    
    for tag in upstream:
        if tag == "regression_summary_mapie":
            continue
        for key in upstream[tag]:
            if files_processed >= max_files:
                break
            # Extract model name
            model_name = key.replace(prefix, "").replace(f"_{ncm_code}", "")
            if datasets is not None and model_name not in datasets:
                logger.info(f"Skip {model_name}")
                continue
            filepath = upstream[tag][key]["results"]
            
            try:
                # Determine metric column name
                metric_col = f"{model_name}_{col_suffix}" if is_classification else col_suffix
                
                # Read entire Excel file (can't chunk Excel natively)
                df = pd.read_excel(
                    filepath,
                    sheet_name="Prediction Intervals",
                    usecols=["ADI", metric_col]
                )
                
                if not is_classification:
                    # Get dataset name from model_name
                    df[f'Relative_{metric_col}'] = df.apply(
                        lambda row: scaler.scale_interval(model_name, row[metric_col]),
                        axis=1
                    )                    
                    df = df.rename(columns={f'Relative_{metric_col}': 'metric'})
                else:
                    # Rename for consistency
                    df = df.rename(columns={metric_col: 'metric'})
                
                # Process in chunks internally to avoid peak memory
                n_rows = len(df)
                for start_idx in range(0, n_rows, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n_rows)
                    chunk = df.iloc[start_idx:end_idx]
                    aggregator.process_dataframe(chunk, model_name)
                
                files_processed += 1
                if files_processed % 10 == 0:
                    logger.info(f"  Processed {files_processed} files...")
                
            except Exception as e:
                files_failed += 1
                logger.error(f"  Failed {model_name}: {str(e)}")
                continue
    
    logger.info(f"\nProcessing complete: {files_processed} files processed, {files_failed} failed")
    logger.info(f"Total predictions: {aggregator.total_chemicals:,}")
    
    # Save cache if path provided
    if cache_path:
        aggregator.save(cache_path)
    
    return aggregator


# -----------------------------
# STATISTICS OUTPUT
# -----------------------------

def write_statistics(aggregator: ConformalAggregator, base_path: str):
    """Write comprehensive statistics to files"""
    
    # Calculate correlation metrics (Distance/Width to ADI - showing robustness)
    mean_metric = aggregator.global_adi.get_mean_metric()
    adi_bin_centers = [0.25, 0.625, 0.8, 0.925]  # Midpoints of ADI bins
    means_for_corr = [mean_metric[k] for k in ADI_LABELS]
    
    # Correlation: metric to ADI (lower uncertainty should predict higher ADI)
    spearman_corr, spearman_p = spearmanr(means_for_corr, adi_bin_centers)
    pearson_corr = np.corrcoef(means_for_corr, adi_bin_centers)[0, 1]
    
    metric_label = METRIC_LABEL
    
    # ===== 1. GLOBAL STATISTICS CSV =====
    global_stats_data = []
    
    std_metric = aggregator.global_adi.get_std_metric()
    quantiles = aggregator.global_adi.get_quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
    
    for label in ADI_LABELS:
        count = aggregator.global_adi.counts[label]
        mean = mean_metric[label]
        std = std_metric[label]
        q05, q25, median, q75, q95 = quantiles[label]
        
        global_stats_data.append({
            'ADI_Bin': label,
            'Count': count,
            'Percentage': 100 * count / aggregator.total_chemicals if aggregator.total_chemicals > 0 else 0,
            f'Mean_{METRIC_NAME}': mean,
            f'Std_{METRIC_NAME}': std,
            'Q05': q05,
            'Q25': q25,
            'Median': median,
            'Q75': q75,
            'Q95': q95
        })
    
    df_global = pd.DataFrame(global_stats_data)
    df_global.to_csv(f"{base_path}_global_stats.csv", index=False, float_format='%.4f')
    logger.info(f"Global statistics saved to: {base_path}_global_stats.csv")
    
    # ===== 2. PER-MODEL STATISTICS CSV =====
    model_stats_data = []
    
    for model_name in aggregator.model_names:
        overall_mean = aggregator.model_all[model_name].mean
        overall_std = aggregator.model_all[model_name].std
        total_count = sum(aggregator.model_adi[model_name].counts.values())
        
        # Get stats by ADI bin for this model
        model_mean_metric = aggregator.model_adi[model_name].get_mean_metric()
        model_std_metric = aggregator.model_adi[model_name].get_std_metric()
        model_quantiles = aggregator.model_adi[model_name].get_quantiles([0.25, 0.5, 0.75])
        
        model_stats_data.append({
            'Model': model_name,
            'Total_Count': total_count,
            'Overall_Mean': overall_mean,
            'Overall_Std': overall_std,
            'VeryLow_Mean': model_mean_metric.get('Very Low', np.nan),
            'VeryLow_Median': model_quantiles.get('Very Low', [np.nan]*3)[1],
            'Low_Mean': model_mean_metric.get('Low', np.nan),
            'Low_Median': model_quantiles.get('Low', [np.nan]*3)[1],
            'Moderate_Mean': model_mean_metric.get('Moderate', np.nan),
            'Moderate_Median': model_quantiles.get('Moderate', [np.nan]*3)[1],
            'High_Mean': model_mean_metric.get('High', np.nan),
            'High_Median': model_quantiles.get('High', [np.nan]*3)[1],
        })
    
    df_models = pd.DataFrame(model_stats_data)
    df_models = df_models.sort_values('Overall_Mean')  # Sort by certainty (low = more certain)
    df_models.to_csv(f"{base_path}_model_stats.csv", index=False, float_format='%.4f')
    logger.info(f"Model statistics saved to: {base_path}_model_stats.csv")
    
    # ===== 3. DETAILED MODEL-ADI MATRIX CSV =====
    model_adi_data = []
    
    for model_name in aggregator.model_names:
        model_mean_metric = aggregator.model_adi[model_name].get_mean_metric()
        model_counts = aggregator.model_adi[model_name].counts
        
        for label in ADI_LABELS:
            model_adi_data.append({
                'Model': model_name,
                'ADI_Bin': label,
                'Count': model_counts[label],
                f'Mean_{METRIC_NAME}': model_mean_metric[label]
            })
    
    df_model_adi = pd.DataFrame(model_adi_data)
    df_model_adi.to_csv(f"{base_path}_model_adi_matrix.csv", index=False, float_format='%.4f')
    logger.info(f"Model-ADI matrix saved to: {base_path}_model_adi_matrix.csv")
    
    # ===== 4. SUMMARY TEXT REPORT =====
    with open(f"{base_path}_summary.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONFORMAL PREDICTION ANALYSIS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Predictions: {aggregator.total_chemicals:,}\n")
        f.write(f"Number of Models: {len(aggregator.model_names)}\n")
        f.write(f"Overall Mean {metric_label}: {aggregator.global_all.mean:.4f}\n")
        f.write(f"Overall Std {metric_label}: {aggregator.global_all.std:.4f}\n\n")
        
        # Correlation metrics
        f.write(f"{metric_label.upper()}-TO-ADI CORRELATION (CP ROBUSTNESS)\n")
        f.write("-"*80 + "\n")
        f.write(f"Spearman rho (rank correlation): {spearman_corr:.4f} (p-value: {spearman_p:.4e})\n")
        f.write(f"Pearson r (linear correlation): {pearson_corr:.4f}\n")
        f.write(f"\nInterpretation:\n")
        if spearman_corr < -0.7:
            f.write("  [EXCELLENT] Strong negative correlation - CP is robust and reliable\n")
            f.write("  Low uncertainty strongly indicates high applicability domain\n")
        elif spearman_corr < -0.5:
            f.write("  [GOOD] Moderate negative correlation - CP is sufficiently robust\n")
            f.write("  Low uncertainty moderately indicates high applicability domain\n")
        elif spearman_corr < 0:
            f.write("  [WEAK] Weak negative correlation - CP shows some robustness\n")
            f.write("  Uncertainty-applicability relationship is present but weak\n")
        else:
            f.write("  [WARNING] No negative correlation - CP may not be robust!\n")
            f.write("  Uncertainty does not properly reflect applicability domain\n")
        f.write("\n")
        
        # Global statistics by ADI
        f.write("GLOBAL STATISTICS BY ADI BIN\n")
        f.write("-"*80 + "\n")
        f.write(f"{'ADI Bin':<15} {'Count':<12} {'%':<8} {'Mean':<10} {'Std':<10} {'Median':<10} {'Q25':<10} {'Q75':<10}\n")
        f.write("-"*80 + "\n")
        
        for label in ADI_LABELS:
            count = aggregator.global_adi.counts[label]
            pct = 100 * count / aggregator.total_chemicals if aggregator.total_chemicals > 0 else 0
            mean = mean_metric[label]
            std = std_metric[label]
            q25, median, q75 = quantiles[label][1], quantiles[label][2], quantiles[label][3]
            
            f.write(f"{label:<15} {count:<12,} {pct:<8.2f} {mean:<10.4f} {std:<10.4f} {median:<10.4f} {q25:<10.4f} {q75:<10.4f}\n")
        
        f.write("\n\n")
        
        # Model rankings
        model_stats = [(name, aggregator.model_all[name].mean, sum(aggregator.model_adi[name].counts.values())) 
                       for name in aggregator.model_names]
        model_stats.sort(key=lambda x: x[1])
        
        f.write(f"MODEL RANKINGS (BY MEAN {metric_label.upper()} - LOWER = MORE CERTAIN)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Model':<50} {'Mean':<15} {'N Predictions':<15}\n")
        f.write("-"*80 + "\n")
        
        for i, (name, mean, count) in enumerate(model_stats, 1):
            f.write(f"{i:<6} {name[:50]:<50} {mean:<15.4f} {count:<15,}\n")
        
        f.write("\n\n")
        
        # Top 10 and Bottom 10
        f.write(f"TOP 10 MOST CERTAIN MODELS (Lowest Mean {metric_label})\n")
        f.write("-"*80 + "\n")
        for i, (name, mean, count) in enumerate(model_stats[:10], 1):
            f.write(f"{i:2d}. {name[:60]:<60} {mean:.4f}\n")
        
        f.write("\n")
        
        f.write(f"TOP 10 MOST UNCERTAIN MODELS (Highest Mean {metric_label})\n")
        f.write("-"*80 + "\n")
        for i, (name, mean, count) in enumerate(reversed(model_stats[-10:]), 1):
            f.write(f"{i:2d}. {name[:60]:<60} {mean:.4f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Summary report saved to: {base_path}_summary.txt")


# -----------------------------
# PLOTTING FUNCTIONS
# -----------------------------

def plot_global_analysis(aggregator: ConformalAggregator, save_path: str):
    """Create comprehensive global analysis plots"""
    
    metric_label = METRIC_LABEL
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Get statistics
    mean_metric = aggregator.global_adi.get_mean_metric()
    std_metric = aggregator.global_adi.get_std_metric()
    
    # ===== Row 1: Summary Statistics =====
    
    # A: Mean metric vs ADI (with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_pos = np.arange(len(ADI_LABELS))
    means = [mean_metric[k] for k in ADI_LABELS]
    stds = [std_metric[k] for k in ADI_LABELS]
    
    ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax1.set_ylabel(f'Mean {metric_label}', fontsize=11, fontweight='bold')
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
    
    # ===== Row 2: Distributions and Correlation =====
    
    # C: Overall metric histogram
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get approximate histogram from global t-digest
    hist_bins = np.linspace(0, aggregator.global_all.mean + 3*aggregator.global_all.std, 50)
    hist_counts = aggregator.global_digest.histogram(hist_bins)
    
    # Histogram as density
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    bin_widths = np.diff(hist_bins)
    total_counts = hist_counts.sum()

    # Convert counts to density
    densities = hist_counts / (total_counts * bin_widths)

    ax3.bar(bin_centers, densities, width=bin_widths, 
            alpha=0.7, color='purple', edgecolor='darkviolet')
    ax3.set_xlabel(metric_label, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title(f'C: Overall {metric_label} Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # D: Metric-to-ADI correlation plot (demonstrating CP robustness)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create scatter data using bin centers and means
    adi_bin_centers = [0.25, 0.625, 0.8, 0.925]  # Midpoints of ADI bins
    means_for_scatter = [mean_metric[k] for k in ADI_LABELS]
    sizes = [aggregator.global_adi.counts[k] / 1000 for k in ADI_LABELS]  # Scale for visibility
    
    # Plot with Metric on Y-axis, ADI on X-axis
    ax4.scatter(adi_bin_centers, means_for_scatter, s=sizes, alpha=0.6, c='steelblue', edgecolors='navy')
    
    # Add trend line
    z = np.polyfit(adi_bin_centers, means_for_scatter, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Calculate Spearman correlation (Metric to ADI)
    spearman_corr, spearman_p = spearmanr(means_for_scatter, adi_bin_centers)
    
    ax4.set_xlabel('ADI (Applicability Domain Index)', fontsize=11, fontweight='bold')
    ax4.set_ylabel(f'Mean {metric_label}', fontsize=11, fontweight='bold')
    ax4.set_title(f'D: CP Robustness Check\nSpearman rho = {spearman_corr:.3f} (p={spearman_p:.4f})', 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim([0, 1])
    
    # Overall title
    task_type = "Classification" if aggregator.is_classification else "Regression"
    fig.suptitle(f'Conformal Prediction Analysis - Global Summary ({task_type})\n({aggregator.total_chemicals:,} predictions across {len(aggregator.model_names)} models)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Global analysis saved to: {save_path}")
    plt.close()


def plot_model_comparison(aggregator: ConformalAggregator, save_path: str, top_n: int = 10):
    """Compare performance across models"""
    
    metric_label = METRIC_LABEL
    
    # Calculate overall mean metric per model
    model_stats = []
    for model_name in aggregator.model_names:
        mean_metrics = aggregator.model_adi[model_name].get_mean_metric()
        overall_mean = aggregator.model_all[model_name].mean
        overall_std = aggregator.model_all[model_name].std
        total_count = sum(aggregator.model_adi[model_name].counts.values())
        model_stats.append((model_name, overall_mean, overall_std, total_count))
    
    # Sort by mean metric
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
    
    # Plot 1: Model ranking by mean metric
    ax1 = fig.add_subplot(gs[0, :])
    models = [s[0] for s in selected]
    means = [s[1] for s in selected]
    stds = [s[2] for s in selected]
    
    y_pos = np.arange(len(models))
    colors = ['green' if i < top_n else 'red' for i in range(len(models))]
    
    ax1.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.6, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlabel(f'Mean {metric_label}', fontsize=11, fontweight='bold')
    ax1.set_title(f'Model Ranking by Uncertainty {title_suffix}', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # Add mean value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(mean + std + 0.1, i, f'{mean:.2f}', va='center', fontsize=8)
    
    # Plot 2: Metric vs ADI for top models
    ax2 = fig.add_subplot(gs[1, 0])
    for model_name in models[:5]:  # Top 5 most certain
        mean_metric = aggregator.model_adi[model_name].get_mean_metric()
        metrics = [mean_metric[k] for k in ADI_LABELS]
        ax2.plot(ADI_LABELS, metrics, marker='o', label=model_name[:20], linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('ADI Bin', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'Mean {metric_label}', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top 5 Models: {metric_label} vs ADI', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    
    # Plot 3: Metric vs ADI for bottom models
    ax3 = fig.add_subplot(gs[1, 1])
    for model_name in models[-5:]:  # Bottom 5 most uncertain
        mean_metric = aggregator.model_adi[model_name].get_mean_metric()
        metrics = [mean_metric[k] for k in ADI_LABELS]
        ax3.plot(ADI_LABELS, metrics, marker='o', label=model_name[:20], linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('ADI Bin', fontsize=11, fontweight='bold')
    ax3.set_ylabel(f'Mean {metric_label}', fontsize=11, fontweight='bold')
    ax3.set_title(f'Bottom 5 Models: {metric_label} vs ADI', fontsize=12, fontweight='bold')
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
    ax4.set_ylabel(metric_label, fontsize=11, fontweight='bold')
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
    logger.info(f"Model comparison saved to: {save_path}")
    plt.close()


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__" or True:  # Works in both script and notebook mode
    
    print("="*60)
    print("CONFORMAL PREDICTION ANALYSIS - STREAMING MODE")
    print("="*60)
    print()
    
    # Set up cache path
    base_path = product["results"].rsplit('.', 1)[0]
    task_suffix = "classification" if classification else "regression"
    cache_path = f"{base_path}_aggregator_cache_{task_suffix}.pkl"

    if classification:
        excluded_models = ["SKIN_IRRITATION", "CARC_SFO_CLASS", "EYE IRRITATION", "EYE_IRRITATION_KNN"]
        scaler = None
    else:
        summary_path = upstream.get(f"{task_suffix}_summary_mapie", {}).get("data", None)
        excluded_models = []
        if summary_path is not None:
            _tmp = pd.read_excel(summary_path)[["Dataset Name", "outlier", "Relative Interval Width"]]
            _tmp["outlier"] = _tmp["outlier"].astype(bool)
            excluded_models = _tmp.loc[_tmp["outlier"]]["Dataset Name"].unique()
            knn_models = _tmp.loc[_tmp["Relative Interval Width"]<0.003]["Dataset Name"].unique()
            print(knn_models)
            excluded_models = np.union1d(excluded_models, knn_models)
            scaler = IntervalScaler.from_summary_file(summary_path)

    # Process all files (will use cache if available)
    aggregator = process_files(upstream, prefix, ncm_code, classification, 
                               max_files=max_files or 300, cache_path=cache_path,
                               scaler=scaler, datasets=data)
    
    aggregator = aggregator.filter_models(excluded_models)
    
    print("\nGenerating outputs...")
    write_statistics(aggregator, base_path)
    plot_global_analysis(aggregator, f"{base_path}.png")
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
    print(f"  - {cache_path} (aggregator cache)")
    print("="*60)
