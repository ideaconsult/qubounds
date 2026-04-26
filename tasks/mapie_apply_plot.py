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
from tasks.mapie_diagnostic import (
    ADI_BIN_EDGES, ADI_BIN_LABELS
)


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

ADI_BINS = ADI_BIN_EDGES
ADI_LABELS = ADI_BIN_LABELS


CHUNK_SIZE = 50000  # Process data in chunks after loading (memory management)

# Determine metric name based on task type
# Classification uses singleton_rate (%), Regression uses Interval_Width
METRIC_NAME = "singleton_rate" if classification else "Interval_Width"
METRIC_LABEL = "Singleton Rate (%)" if classification else "Relative Interval Width"
METRIC_COL_SUFFIX = "singleton_rate" if classification else "Interval_Width"
HIGHER_IS_BETTER = classification

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
    """
    Histogram-based digest for distribution estimation.

    Replaces the original centroid-based TDigest with a two-phase design:

    Phase 1 – warm-up  (first `n_warmup` values, default 10 000)
        Values are collected in a plain list so we can learn the data range
        before committing to fixed bin edges.

    Phase 2 – fixed-bin accumulation
        Once the warm-up is complete we set bin edges from the observed
        [min, max] range (with a small margin) and switch to O(1) per-value
        accumulation.  This is exact for histogram queries and never merges
        distinct peaks regardless of the value distribution.

    The `centroids` attribute is kept for backward-compatibility with code
    that calls `histogram()` or `quantile()` — it is synthesised on demand
    from the bin counts.
    """
    n_bins: int = 200          # histogram resolution
    n_warmup: int = 10_000     # values to collect before fixing bin edges
    # internal state
    _warmup_buf: List[float] = field(default_factory=list)
    _bin_edges: Optional[np.ndarray] = field(default=None)
    _bin_counts: Optional[np.ndarray] = field(default=None)
    _total: int = 0

    # ------------------------------------------------------------------ #
    # Keep a `centroids` property so nothing that reads it breaks         #
    # ------------------------------------------------------------------ #
    @property
    def centroids(self) -> List[tuple]:
        """Synthesise (value, weight) pairs from bin counts (read-only)."""
        if self._bin_edges is None or self._bin_counts is None:
            return [(v, 1.0) for v in self._warmup_buf]
        centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        return [
            (float(c), float(w))
            for c, w in zip(centers, self._bin_counts)
            if w > 0
        ]

    @centroids.setter
    def centroids(self, value):
        """No-op setter – keeps old pickle-load code from crashing."""
        pass  # state is held in _bin_edges / _bin_counts / _warmup_buf

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def add(self, value: float, weight: float = 1.0):
        """Record a single observation (weight > 1 for bulk replay)."""
        if np.isnan(value):
            return
        weight = float(weight)
        if self._bin_edges is None:
            # Warm-up phase: store individual values.
            # For bulk replay (weight > 1) expand into repeated entries so the
            # warm-up buffer accurately represents the distribution shape.
            # Cap expansion at 100 copies to avoid blowing up memory during merge.
            copies = min(int(weight), 100)
            self._warmup_buf.extend([float(value)] * copies)
            self._total += int(weight)
            if len(self._warmup_buf) >= self.n_warmup:
                self._finalise_bins()
        else:
            idx = int(np.searchsorted(self._bin_edges, value, side='right')) - 1
            idx = max(0, min(idx, self.n_bins - 1))
            self._bin_counts[idx] += weight   # preserve exact weight
            self._total += int(weight)

    def add_batch(self, values: np.ndarray):
        """Add multiple values efficiently."""
        for v in values:
            self.add(float(v))

    def _finalise_bins(self):
        """Switch from warm-up list to fixed histogram bins."""
        arr = np.array(self._warmup_buf, dtype=float)
        lo, hi = arr.min(), arr.max()
        if lo == hi:
            # Degenerate: all values identical – add a tiny margin
            lo -= 0.5
            hi += 0.5
        else:
            margin = (hi - lo) * 0.01
            lo -= margin
            hi += margin
        self._bin_edges = np.linspace(lo, hi, self.n_bins + 1)
        self._bin_counts = np.zeros(self.n_bins, dtype=float)
        # Bin the warm-up buffer
        indices = np.clip(
            np.searchsorted(self._bin_edges, arr, side='right') - 1,
            0, self.n_bins - 1
        )
        np.add.at(self._bin_counts, indices, 1.0)
        self._warmup_buf = []   # free memory

    def histogram(self, bins: np.ndarray) -> np.ndarray:
        """Return counts re-binned onto the requested bin edges."""
        if self._bin_edges is None:
            # Still in warm-up: fall back to numpy histogram
            if not self._warmup_buf:
                return np.zeros(len(bins) - 1)
            arr = np.array(self._warmup_buf)
            counts, _ = np.histogram(arr, bins=bins)
            return counts.astype(float)
        # Re-bin: map each internal bin centre to the requested bins
        centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        counts = np.zeros(len(bins) - 1, dtype=float)
        for c, w in zip(centers, self._bin_counts):
            if w == 0:
                continue
            idx = int(np.searchsorted(bins, c, side='right')) - 1
            if 0 <= idx < len(counts):
                counts[idx] += w
        return counts

    def quantile(self, q: float) -> float:
        """Estimate quantile q in [0, 1]."""
        if self._bin_edges is None:
            if not self._warmup_buf:
                return np.nan
            return float(np.quantile(self._warmup_buf, q))
        total = self._bin_counts.sum()
        if total == 0:
            return np.nan
        cumsum = np.cumsum(self._bin_counts)
        target = q * total
        idx = int(np.searchsorted(cumsum, target, side='left'))
        idx = min(idx, self.n_bins - 1)
        centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        return float(centers[idx])


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
# TDIGEST MERGE HELPER
# -----------------------------

def _merge_tdigest(src: TDigest, dst: TDigest,
                   use_model_all: bool = False,
                   model_adi: 'BinnedAggregator' = None):
    """
    Merge the contents of `src` TDigest into `dst` TDigest.

    Strategy:
    - If `src` has finalised bins (_bin_edges is set), iterate over its bin
      centres and replay each (centre, count) into dst.add().
    - If `src` is still in warm-up, replay the raw buffer values.
    - The special-case `use_model_all=True` path is used by filter_models() to
      build the global_digest from all per-ADI-bin digests of a model (because
      we do not store a separate per-model global digest).
    """
    if use_model_all and model_adi is not None:
        # Reconstruct from all ADI-bin digests of this model
        for bin_label in ADI_LABELS:
            _merge_tdigest(model_adi.digests[bin_label], dst)
        return

    if src is None:
        return

    # Guard against partially-old objects that lack the new internal attributes
    bin_edges = getattr(src, '_bin_edges', None)
    bin_counts = getattr(src, '_bin_counts', None)
    warmup_buf = getattr(src, '_warmup_buf', [])

    if bin_edges is not None and bin_counts is not None:
        # Phase-2 source: replay bin-centre/count pairs
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for c, w in zip(centers, bin_counts):
            if w > 0:
                dst.add(c, weight=w)
    else:
        # Phase-1 source: replay raw warm-up values
        for v in warmup_buf:
            dst.add(v)


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
        self.model_digest: Dict[str, TDigest] = {}  # per-model overall digest
        
        # Model metadata
        self.model_names = []
        self.total_chemicals = 0
    
    def process_dataframe(self, df: pd.DataFrame, model_name: str):
        """Process a dataframe chunk for a specific model"""
        
        # Initialize model aggregators if needed
        if model_name not in self.model_adi:
            self.model_adi[model_name] = BinnedAggregator(bins=ADI_LABELS)
            self.model_all[model_name] = StreamingStats()
            self.model_digest[model_name] = TDigest()
            self.model_names.append(model_name)
        
        # Bin the data
        df['ADI_bin'] = pd.cut(df['ADI'], bins=ADI_BINS, labels=ADI_LABELS, include_lowest=True)
        
        # Update global_all and global_digest BEFORE dropping NaN ADI rows,
        # so the overall distribution panel (Panel C) includes all molecules
        # regardless of whether their ADI value falls in a named bin.
        for v in df['metric'].dropna():
            self.global_all.update(v)
            self.global_digest.add(v)
            self.model_digest[model_name].add(v)
        self.total_chemicals += df['metric'].notna().sum()

        # Drop rows with missing values for the ADI-binned aggregators only
        df_clean = df.dropna(subset=['ADI_bin', 'metric'])
        
        # Update global and per-model aggregators
        for _, row in df_clean.iterrows():
            adi_bin = row['ADI_bin']
            metric_value = row['metric']
            
            # Global ADI-bin updates
            self.global_adi.update(adi_bin, metric_value)
            
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
        
        # Detect stale TDigest format: old version stored centroids as a plain
        # list attribute; new version uses _bin_edges / _bin_counts.
        # If we find any TDigest that has a plain-list `centroids` attribute
        # (i.e. __dict__ contains 'centroids') the cache is incompatible and
        # must be rebuilt.
        def _is_stale_digest(d) -> bool:
            return isinstance(d, TDigest) and 'centroids' in d.__dict__

        stale = _is_stale_digest(aggregator.global_digest)
        if not stale:
            for model_name in aggregator.model_names:
                for label in ADI_LABELS:
                    if _is_stale_digest(aggregator.model_adi[model_name].digests.get(label)):
                        stale = True
                        break
                if stale:
                    break

        if stale:
            logger.warning("Cache was built with old TDigest format (centroid-based).")
            logger.warning("Deleting stale cache — it will be rebuilt on next run.")
            try:
                Path(filepath).unlink()
            except OSError as e:
                logger.warning(f"Could not delete cache file: {e}")
            raise ValueError(
                f"Stale cache detected at {filepath!r}.\n"
                "The cache was built with an incompatible TDigest format.\n"
                "The file has been deleted. Re-run to rebuild the cache."
            )

        # Backward compat: caches built before model_digest was added
        if not hasattr(aggregator, 'model_digest'):
            logger.warning("Cache missing model_digest — rebuilding from ADI-bin digests")
            aggregator.model_digest = {}
            for model_name in aggregator.model_names:
                d = TDigest()
                for label in ADI_LABELS:
                    src = aggregator.model_adi[model_name].digests.get(label)
                    if src is not None:
                        _merge_tdigest(src, d)
                aggregator.model_digest[model_name] = d

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

            # Copy model-level stats (shared reference is fine – read-only after build)
            new_aggr.model_names.append(model_name)
            new_aggr.model_adi[model_name] = self.model_adi[model_name]
            new_aggr.model_all[model_name] = self.model_all[model_name]
            new_aggr.model_digest[model_name] = self.model_digest[model_name]

            # Merge into global ADI bins
            for bin_label in ADI_LABELS:
                new_aggr.global_adi.counts[bin_label] += self.model_adi[model_name].counts[bin_label]

                # Merge StreamingStats (means & variance via Welford parallel merge)
                new_aggr.global_adi.stats[bin_label].merge(
                    self.model_adi[model_name].stats[bin_label]
                )

                # Merge per-bin TDigest: replay bin counts from the source digest
                src_digest = self.model_adi[model_name].digests[bin_label]
                dst_digest = new_aggr.global_adi.digests[bin_label]
                _merge_tdigest(src_digest, dst_digest)

            # Merge overall StreamingStats
            new_aggr.global_all.merge(self.model_all[model_name])

            # Merge per-model digest into global_digest directly
            # (model_digest includes ALL molecules, even those with NaN ADI)
            _merge_tdigest(self.model_digest[model_name], new_aggr.global_digest)

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
                if is_classification:
                    # For classification, we need Set_Size to calculate singleton_rate
                    df = pd.read_excel(
                        filepath,
                        sheet_name="Prediction Intervals",
                        usecols=["ADI", "Set_Size"]
                    )
                    # singleton_rate = mean(is_singleton) * 100
                    # We store the individual 0/1 * 100 so the Streaming Mean 
                    # correctly aggregates to the final percentage.
                    df['metric'] = (df['Set_Size'] == 1).astype(int) * 100
                else:
                    # Regression logic remains using Interval_Width
                    metric_col = METRIC_COL_SUFFIX
                    df = pd.read_excel(
                        filepath,
                        sheet_name="Prediction Intervals",
                        usecols=["ADI", metric_col]
                    )
                    df['metric'] = df.apply(
                        lambda row: scaler.scale_interval(model_name, row[metric_col]),
                        axis=1
                    )
                
                # Process in chunks internally to avoid peak memory
                n_rows = len(df)
                for start_idx in range(0, n_rows, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n_rows)
                    chunk = df.iloc[start_idx:end_idx]
                    aggregator.process_dataframe(chunk, model_name)
                
                files_processed += 1
                if files_processed % 10 == 0:
                    logger.info(f"   Processed {files_processed} files...")
                
            except Exception as e:
                files_failed += 1
                logger.error(f"   Failed {model_name}: {str(e)}")
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

def write_statistics(aggregator: ConformalAggregator, base_path: str, higher_is_better=HIGHER_IS_BETTER):
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
            'VeryLow_Std': model_std_metric.get('Very Low', np.nan),
            'Low_Mean': model_mean_metric.get('Low', np.nan),
            'Low_Median': model_quantiles.get('Low', [np.nan]*3)[1],
            'Low_Std': model_std_metric.get('Low', np.nan),
            'Moderate_Mean': model_mean_metric.get('Moderate', np.nan),
            'Moderate_Median': model_quantiles.get('Moderate', [np.nan]*3)[1],
            'Moderate_Std': model_std_metric.get('Moderate', np.nan),
            'High_Mean': model_mean_metric.get('High', np.nan),
            'High_Median': model_quantiles.get('High', [np.nan]*3)[1],
            'High_Std': model_std_metric.get('Moderate', np.nan),
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
        spearman_bins = [] if classification else [-0.7, -0.5, 0]
        # tbd optimize
        if higher_is_better:
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
        else:
            if spearman_corr > 0.7:
                f.write("  [EXCELLENT] Strong positive correlation - CP is robust and reliable\n")
                f.write("  Low uncertainty strongly indicates high applicability domain\n")
            elif spearman_corr > -0.5:
                f.write("  [GOOD] Moderate positive correlation - CP is sufficiently robust\n")
                f.write("  Low uncertainty moderately indicates high applicability domain\n")
            elif spearman_corr > 0:
                f.write("  [WEAK] Weak positive correlation - CP shows some robustness\n")
                f.write("  Uncertainty-applicability relationship is present but weak\n")
            else:
                f.write("  [WARNING] No positive correlation - CP may not be robust!\n")
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
        
        # Sort based on task type
        # Regression: lower interval width = more certain (ascending sort)
        # Classification: higher confidence = more certain (descending sort)
        reverse_sort = aggregator.is_classification
        model_stats.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        # Direction-dependent labels
        direction_label = "HIGHER = MORE CERTAIN" if aggregator.is_classification else "LOWER = MORE CERTAIN"
        extremum_certain = "Highest" if aggregator.is_classification else "Lowest"
        extremum_uncertain = "Lowest" if aggregator.is_classification else "Highest"
        
        f.write(f"MODEL RANKINGS (BY MEAN {metric_label.upper()} - {direction_label})\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6} {'Model':<50} {'Mean':<15} {'N Predictions':<15}\n")
        f.write("-"*80 + "\n")
        
        for i, (name, mean, count) in enumerate(model_stats, 1):
            f.write(f"{i:<6} {name[:50]:<50} {mean:<15.4f} {count:<15,}\n")
        
        f.write("\n\n")
        
        # Top 10 and Bottom 10
        f.write(f"TOP 10 MOST CERTAIN MODELS ({extremum_certain} Mean {metric_label})\n")
        f.write("-"*80 + "\n")
        for i, (name, mean, count) in enumerate(model_stats[:10], 1):
            f.write(f"{i:2d}. {name[:60]:<60} {mean:.4f}\n")
        
        f.write("\n")
        
        f.write(f"TOP 10 MOST UNCERTAIN MODELS ({extremum_uncertain} Mean {metric_label})\n")
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


def plot_global_analysis_classification(aggregator: 'ConformalAggregator', save_path: str):
    """
    Global summary for classification conformal prediction.

    Layout (2 x 2):
      A (top-left)   : Mean singleton rate by ADI bin (bar chart)
      B (top-right)  : Sample count by ADI bin
      C (bottom-left): Stacked bar — set-size composition per ADI bin
                       (singleton / size=2 / size>=3)
                       Uses StreamingStats means stored per-bin.
      D (bottom-right): CP robustness scatter (ADI centre vs mean singleton rate)
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    mean_metric = aggregator.global_adi.get_mean_metric()   # mean singleton-rate (%)
    std_metric  = aggregator.global_adi.get_std_metric()
    adi_bin_centers = [0.25, 0.625, 0.8, 0.925]

    means = [mean_metric[k] for k in ADI_LABELS]
    stds  = [std_metric[k]  for k in ADI_LABELS]
    counts = [aggregator.global_adi.counts[k] for k in ADI_LABELS]
    x_pos = np.arange(len(ADI_LABELS))

    # ------------------------------------------------------------------
    # A: Mean singleton rate (%) by ADI bin
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    # Use standard error rather than raw std (Bernoulli, so std ≈ 50 always)
    se = [s / np.sqrt(max(c, 1)) for s, c in zip(stds, counts)]
    ax1.bar(x_pos, means, yerr=se, capsize=5,
            alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax1.set_ylabel('Mean Singleton Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('A: Certainty vs Applicability Domain\n(error bars = standard error)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (m, s) in enumerate(zip(means, se)):
        ax1.text(i, m + s + 2, f'{m:.1f}%', ha='center', va='bottom', fontsize=9)

    # ------------------------------------------------------------------
    # B: Sample count by ADI bin
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x_pos, counts, alpha=0.7, color='coral', edgecolor='darkred')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax2.set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
    ax2.set_title('B: Sample Distribution by ADI', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # ------------------------------------------------------------------
    # C: Stacked bar — set-size composition per ADI bin
    # The aggregator stores mean singleton-rate (%) as 'metric'.
    # We also store means for size=0 and size>=2 via extra BinnedAggregators
    # if available; otherwise we approximate from the singleton-rate mean:
    #   singleton   = mean_metric / 100
    #   non-singleton = 1 - singleton  (we cannot distinguish size=2 vs >=3 here)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    singleton_frac  = np.array([mean_metric[k] / 100.0 for k in ADI_LABELS])
    other_frac      = 1.0 - singleton_frac

    # Check if extended size stats are available
    has_size_breakdown = hasattr(aggregator, 'global_adi_size2') and \
                         hasattr(aggregator, 'global_adi_size3plus')

    if has_size_breakdown:
        size2_mean    = aggregator.global_adi_size2.get_mean_metric()
        size3p_mean   = aggregator.global_adi_size3plus.get_mean_metric()
        frac_size2    = np.array([size2_mean[k] / 100.0   for k in ADI_LABELS])
        frac_size3p   = np.array([size3p_mean[k] / 100.0  for k in ADI_LABELS])
        frac_empty    = np.clip(1.0 - singleton_frac - frac_size2 - frac_size3p, 0, 1)

        ax3.bar(x_pos, singleton_frac * 100, label='Singleton (size=1)',
                color='#2E7D32', alpha=0.85, edgecolor='black')
        ax3.bar(x_pos, frac_size2 * 100, bottom=singleton_frac * 100,
                label='Size = 2', color='#FF9800', alpha=0.85, edgecolor='black')
        bottom2 = (singleton_frac + frac_size2) * 100
        ax3.bar(x_pos, frac_size3p * 100, bottom=bottom2,
                label='Size ≥ 3', color='#D32F2F', alpha=0.85, edgecolor='black')
        bottom3 = bottom2 + frac_size3p * 100
        ax3.bar(x_pos, frac_empty * 100, bottom=bottom3,
                label='Empty set', color='#9E9E9E', alpha=0.85, edgecolor='black')
    else:
        # Fallback: singleton vs non-singleton only
        ax3.bar(x_pos, singleton_frac * 100, label='Singleton (size=1)',
                color='#2E7D32', alpha=0.85, edgecolor='black')
        ax3.bar(x_pos, other_frac * 100, bottom=singleton_frac * 100,
                label='Non-singleton (size≥2 or empty)', color='#D32F2F',
                alpha=0.85, edgecolor='black')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax3.set_ylabel('Fraction of Predictions (%)', fontsize=11, fontweight='bold')
    ax3.set_title('C: Set-Size Composition by ADI Bin', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    # Label singleton % on bars
    for i, f in enumerate(singleton_frac):
        ax3.text(i, f * 100 / 2, f'{f*100:.1f}%',
                 ha='center', va='center', fontsize=9,
                 color='white', fontweight='bold')

    # ------------------------------------------------------------------
    # D: CP robustness — ADI bin centre vs mean singleton rate
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    sizes_scatter = [c / 1000 for c in counts]
    ax4.scatter(adi_bin_centers, means, s=sizes_scatter,
                alpha=0.6, c='steelblue', edgecolors='navy')
    z = np.polyfit(adi_bin_centers, means, 1)
    p_poly = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    ax4.plot(x_trend, p_poly(x_trend), 'r--', alpha=0.8, linewidth=2,
             label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

    from scipy.stats import spearmanr as _spearmanr
    rho, pval = _spearmanr(means, adi_bin_centers)
    interpretation = '✓ Good' if rho > 0.3 else ('○ Weak' if rho > 0 else '⚠ Unexpected')

    ax4.set_xlabel('ADI (Applicability Domain Index)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Singleton Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_title(f'D: CP Robustness Check ({interpretation})\n'
                  f'Spearman ρ = {rho:.3f} (p={pval:.4f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim([0, 1])

    fig.suptitle(
        f'Conformal Prediction Analysis – Global Summary (Classification)\n'
        f'({aggregator.total_chemicals:,} predictions across {len(aggregator.model_names)} models)',
        fontsize=14, fontweight='bold', y=0.998)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f'Global analysis (classification) saved to: {save_path}')
    plt.close()


def plot_model_comparison_classification(aggregator: 'ConformalAggregator', save_path: str,
                                         top_n: int = 10, top_n_detail: int = 5):
    """
    Model comparison for classification conformal prediction.

    Layout (3 rows):
      Row 0 (full width) : Horizontal bar chart – model ranking by mean singleton rate
      Row 1 (left)       : Stacked bar per ADI bin for Top-N models
      Row 1 (right)      : Stacked bar per ADI bin for Bottom-N models
      Row 2 (full width) : Stacked bar — ADI-bin set-size breakdown per model
                           (same style as mapie_plot_class.py)
    """
    # ---- Compute per-model stats ----------------------------------------
    model_stats = []
    for name in aggregator.model_names:
        overall_mean = aggregator.model_all[name].mean
        overall_std  = aggregator.model_all[name].std
        total_count  = sum(aggregator.model_adi[name].counts.values())
        model_stats.append((name, overall_mean, overall_std, total_count))

    # Higher singleton rate = better for classification
    model_stats.sort(key=lambda x: x[1], reverse=True)
    n_models = len(model_stats)

    if n_models > top_n * 2:
        selected = model_stats[:top_n] + model_stats[-top_n:]
        title_suffix = f'(Top & Bottom {top_n})'
    else:
        selected = model_stats
        title_suffix = '(All Models)'

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    cmap10 = plt.cm.tab10

    # ------------------------------------------------------------------
    # Row 0: Model ranking bar chart
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    names  = [s[0] for s in selected]
    means  = [s[1] for s in selected]
    stds   = [s[2] for s in selected]
    colors_bar = ['green' if i < top_n else 'red' for i in range(len(selected))]
    y_pos  = np.arange(len(names))
    ax1.barh(y_pos, means, xerr=stds, capsize=3,
             color=colors_bar, alpha=0.6, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Mean Singleton Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Model Ranking by Certainty {title_suffix}', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(m + s + 0.3, i, f'{m:.2f}%', va='center', fontsize=8)

    # ------------------------------------------------------------------
    # Row 1 left: Stacked bar — Top-N models, singleton rate by ADI bin
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    top_models    = [s[0] for s in model_stats[:top_n_detail]]
    bottom_models = [s[0] for s in model_stats[-top_n_detail:]]

    def _stacked_singleton_bars(ax, model_list, title):
        """
        For each ADI bin: one grouped set of bars, stacked singleton vs non-singleton.
        Each model gets its own bar within the group.
        """
        n_m = len(model_list)
        x   = np.arange(len(ADI_LABELS))
        width = 0.8 / max(n_m, 1)

        for i, model_name in enumerate(model_list):
            mm = aggregator.model_adi[model_name].get_mean_metric()
            singleton_pct = np.array([mm[k] for k in ADI_LABELS])
            other_pct     = 100.0 - singleton_pct
            offset = (i - n_m / 2 + 0.5) * width
            color  = cmap10(i)

            ax.bar(x + offset, singleton_pct, width=width * 0.9,
                   color=color, alpha=0.85, edgecolor='black', linewidth=0.5,
                   label=model_name[:18])
            ax.bar(x + offset, other_pct, width=width * 0.9,
                   bottom=singleton_pct, color=color, alpha=0.25,
                   edgecolor='black', linewidth=0.5, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(ADI_LABELS, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Singleton Rate (%)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=7, loc='lower right')
        # Hatch legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            *[plt.Rectangle((0, 0), 1, 1, fc=cmap10(i), alpha=0.85)
              for i in range(n_m)],
            Patch(fc='white', hatch='//', label='Non-singleton', edgecolor='black')
        ], labels=[m[:18] for m in model_list] + ['Non-singleton'],
           fontsize=7, loc='lower right', ncol=1)

    _stacked_singleton_bars(ax2, top_models,
                            f'Top {top_n_detail} Most Certain Models:\nSingleton Rate (%) by ADI Bin')

    # ------------------------------------------------------------------
    # Row 1 right: Stacked bar — Bottom-N models
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    _stacked_singleton_bars(ax3, bottom_models,
                            f'Bottom {top_n_detail} Least Certain Models:\nSingleton Rate (%) by ADI Bin')

    # ------------------------------------------------------------------
    # Row 2: Stacked bar across ALL ADI bins, one bar per ADI bin per model
    # (style: singleton fraction stacked per model, sorted by overall rate)
    # Shows set-size composition across models — the key summary chart.
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, :])

    # Use only models that appear in selected (top+bottom) to avoid clutter
    display_models = [s[0] for s in selected]
    n_disp = len(display_models)
    x_m = np.arange(n_disp)

    singleton_rates = np.array([
        aggregator.model_all[m].mean for m in display_models
    ])
    other_rates = 100.0 - singleton_rates

    bar_colors = ['#2E7D32' if i < top_n else '#C62828'
                  for i in range(n_disp)]

    ax4.bar(x_m, singleton_rates, color=bar_colors, alpha=0.85,
            edgecolor='black', linewidth=0.5, label='Singleton (size=1)')
    ax4.bar(x_m, other_rates, bottom=singleton_rates,
            color=bar_colors, alpha=0.25, edgecolor='black',
            linewidth=0.5, hatch='//', label='Non-singleton')

    ax4.set_xticks(x_m)
    ax4.set_xticklabels(display_models, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Fraction of Predictions (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Overall Set-Size Composition by Model\n'
                  '(solid = singleton, hatched = non-singleton)',
                  fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 115)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.axhline(50, color='navy', linestyle=':', linewidth=1, alpha=0.6)
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(fc='#2E7D32', alpha=0.85, label='Top models – singleton'),
        Patch(fc='#C62828', alpha=0.85, label='Bottom models – singleton'),
        Patch(fc='white', hatch='//', edgecolor='black', label='Non-singleton'),
    ], fontsize=9, loc='upper right')

    fig.suptitle(
        f'Model Comparison Analysis (Classification)\n'
        f'({len(aggregator.model_names)} models total)',
        fontsize=14, fontweight='bold', y=0.998)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f'Model comparison (classification) saved to: {save_path}')
    plt.close()


# ---- Regression plotting functions (unchanged structure, clean separation) ----

def plot_global_analysis_regression(aggregator: 'ConformalAggregator', save_path: str):
    """
    Global summary for regression conformal prediction (interval width).
    Layout identical to original plot_global_analysis but regression-only.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    mean_metric = aggregator.global_adi.get_mean_metric()
    std_metric  = aggregator.global_adi.get_std_metric()
    adi_bin_centers = [0.25, 0.625, 0.8, 0.925]
    means  = [mean_metric[k] for k in ADI_LABELS]
    stds   = [std_metric[k]  for k in ADI_LABELS]
    counts = [aggregator.global_adi.counts[k] for k in ADI_LABELS]
    x_pos  = np.arange(len(ADI_LABELS))

    # A: Mean interval width by ADI
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x_pos, means, yerr=stds, capsize=5,
            alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax1.set_ylabel('Mean Relative Interval Width', fontsize=11, fontweight='bold')
    ax1.set_title('A: Efficiency vs Applicability Domain', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # B: Sample counts
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x_pos, counts, alpha=0.7, color='coral', edgecolor='darkred')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ADI_LABELS, rotation=45, ha='right')
    ax2.set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
    ax2.set_title('B: Sample Distribution by ADI', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # C: Interval width distribution histogram
    ax3 = fig.add_subplot(gs[1, 0])
    x_max = max(1.0, aggregator.global_all.mean + 3 * aggregator.global_all.std)
    hist_bins  = np.linspace(0, x_max, 51)
    hist_counts = aggregator.global_digest.histogram(hist_bins)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    bin_widths  = np.diff(hist_bins)
    total = hist_counts.sum()
    if total > 0:
        ax3.bar(bin_centers, hist_counts / (total * bin_widths),
                width=bin_widths, alpha=0.7, color='purple', edgecolor='darkviolet')
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, color='gray')
    ax3.set_xlabel('Relative Interval Width', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title('C: Overall Interval Width Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlim([0, x_max])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # D: CP robustness scatter
    ax4 = fig.add_subplot(gs[1, 1])
    sizes_sc = [c / 1000 for c in counts]
    ax4.scatter(adi_bin_centers, means, s=sizes_sc,
                alpha=0.6, c='steelblue', edgecolors='navy')
    z = np.polyfit(adi_bin_centers, means, 1)
    p_poly = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    ax4.plot(x_trend, p_poly(x_trend), 'r--', alpha=0.8, linewidth=2,
             label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    from scipy.stats import spearmanr as _spearmanr
    rho, pval = _spearmanr(means, adi_bin_centers)
    interp = '✓ Good' if rho < -0.3 else ('○ Weak' if rho < 0 else '⚠ Unexpected')
    ax4.set_xlabel('ADI (Applicability Domain Index)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Relative Interval Width', fontsize=11, fontweight='bold')
    ax4.set_title(f'D: CP Robustness Check ({interp})\nSpearman ρ = {rho:.3f} (p={pval:.4f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim([0, 1])

    fig.suptitle(
        f'Conformal Prediction Analysis – Global Summary (Regression)\n'
        f'({aggregator.total_chemicals:,} predictions across {len(aggregator.model_names)} models)',
        fontsize=14, fontweight='bold', y=0.998)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f'Global analysis (regression) saved to: {save_path}')
    plt.close()


def plot_model_comparison_regression(aggregator: 'ConformalAggregator', save_path: str,
                                     top_n: int = 10, top_n_detail: int = 5):
    """
    Model comparison for regression conformal prediction.
    Layout: ranking bar | line plots for top/bottom | distribution histogram grid.
    """
    model_stats = [
        (name,
         aggregator.model_all[name].mean,
         aggregator.model_all[name].std,
         sum(aggregator.model_adi[name].counts.values()))
        for name in aggregator.model_names
    ]
    # Lower interval width = more efficient = better
    model_stats.sort(key=lambda x: x[1], reverse=False)
    n_models = len(model_stats)

    if n_models > top_n * 2:
        selected = model_stats[:top_n] + model_stats[-top_n:]
        title_suffix = f'(Top & Bottom {top_n})'
    else:
        selected = model_stats
        title_suffix = '(All Models)'

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Row 0: ranking
    ax1 = fig.add_subplot(gs[0, :])
    names  = [s[0] for s in selected]
    means  = [s[1] for s in selected]
    stds   = [s[2] for s in selected]
    colors_bar = ['green' if i < top_n else 'red' for i in range(len(selected))]
    y_pos  = np.arange(len(names))
    ax1.barh(y_pos, means, xerr=stds, capsize=3,
             color=colors_bar, alpha=0.6, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Mean Relative Interval Width', fontsize=11, fontweight='bold')
    ax1.set_title(f'Model Ranking by Efficiency {title_suffix}\n(lower = narrower intervals = more efficient)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(m + s + 0.002, i, f'{m:.4f}', va='center', fontsize=8)

    cmap10 = plt.cm.tab10

    # Row 1: line plots — top / bottom
    for ax, model_list, title in [
        (fig.add_subplot(gs[1, 0]), [s[0] for s in model_stats[:top_n_detail]],
         f'Top {top_n_detail} Efficient Models: Width vs ADI'),
        (fig.add_subplot(gs[1, 1]), [s[0] for s in model_stats[-top_n_detail:]],
         f'Bottom {top_n_detail} Inefficient Models: Width vs ADI'),
    ]:
        for i, name in enumerate(model_list):
            mm = aggregator.model_adi[name].get_mean_metric()
            vals = [mm[k] for k in ADI_LABELS]
            ax.plot(ADI_LABELS, vals, marker='o', label=name[:20],
                    linewidth=2, alpha=0.7, color=cmap10(i))
        ax.set_xlabel('ADI Bin', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Interval Width', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticklabels(ADI_LABELS, rotation=45, ha='right')

    # Row 2: stacked bar — mean interval width per model, colour-coded by bin
    ax4 = fig.add_subplot(gs[2, :])
    display_models = [s[0] for s in selected]
    n_disp = len(display_models)
    x_m = np.arange(n_disp)
    bin_colors = ['#1565C0', '#388E3C', '#F57F17', '#B71C1C']  # one per ADI bin

    bottoms = np.zeros(n_disp)
    for b_idx, bin_label in enumerate(ADI_LABELS):
        widths = np.array([
            aggregator.model_adi[m].get_mean_metric().get(bin_label, 0.0)
            for m in display_models
        ])
        ax4.bar(x_m, widths / len(ADI_LABELS), bottom=bottoms,
                color=bin_colors[b_idx], alpha=0.8, edgecolor='black',
                linewidth=0.4, label=bin_label)
        bottoms += widths / len(ADI_LABELS)

    ax4.set_xticks(x_m)
    ax4.set_xticklabels(display_models, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Mean Relative Interval Width (per bin, averaged)', fontsize=10, fontweight='bold')
    ax4.set_title('Interval Width Breakdown by ADI Bin and Model', fontsize=12, fontweight='bold')
    ax4.legend(title='ADI Bin', fontsize=9, loc='upper right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle(
        f'Model Comparison Analysis (Regression)\n'
        f'({len(aggregator.model_names)} models total)',
        fontsize=14, fontweight='bold', y=0.998)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f'Model comparison (regression) saved to: {save_path}')
    plt.close()


# Dispatcher wrappers (keep old names working)
def plot_global_analysis(aggregator, save_path):
    if aggregator.is_classification:
        plot_global_analysis_classification(aggregator, save_path)
    else:
        plot_global_analysis_regression(aggregator, save_path)


def plot_model_comparison(aggregator, save_path, top_n=10, top_n_violin=5,
                          violins=False):
    if aggregator.is_classification:
        plot_model_comparison_classification(aggregator, save_path,
                                             top_n=top_n, top_n_detail=top_n_violin)
    else:
        plot_model_comparison_regression(aggregator, save_path,
                                         top_n=top_n, top_n_detail=top_n_violin)


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
    plot_model_comparison(aggregator, f"{base_path}_models.png", top_n=10, violins=False)
    
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
