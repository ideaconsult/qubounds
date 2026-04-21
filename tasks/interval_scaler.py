"""
Interval Scaling Utility for Regression Conformal Prediction

This module provides functionality to scale prediction intervals by the training set range.
For regression models, interval widths should be normalized by (max - min) of the training data
to make uncertainty comparable across different datasets and scales.

Usage:
    scaler = IntervalScaler.from_summary_file('path/to/summary.xlsx')
    scaled_value = scaler.scale_interval(dataset_name='MY_DATASET', interval_width=0.5)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class IntervalScaler:
    """
    Scale prediction intervals by training set range for regression models.
    
    Attributes:
        scaling_factors (Dict[str, float]): Mapping of dataset name to (max - min) range
    """
    
    def __init__(self, scaling_factors: Dict[str, float]):
        """
        Initialize the interval scaler.
        
        Args:
            scaling_factors: Dictionary mapping dataset names to their training range (max - min)
        """
        self.scaling_factors = scaling_factors
        logger.info(f"Initialized IntervalScaler with {len(scaling_factors)} datasets")
    
    @classmethod
    def from_summary_file(cls, summary_path: str) -> 'IntervalScaler':
        """
        Load scaling factors from summary Excel file.
        
        The summary file should contain columns:
        - 'Dataset Name': Name of the dataset/model
        - 'Split': Type of split (Training, Test, etc.)
        - 'Min': Minimum value in the dataset
        - 'Max': Maximum value in the dataset
        
        Args:
            summary_path: Path to summary.xlsx file
            
        Returns:
            IntervalScaler instance with loaded scaling factors
            
        Raises:
            FileNotFoundError: If summary file doesn't exist
            ValueError: If required columns are missing
            KeyError: If no Training split found for any dataset
        """
        summary_path = Path(summary_path)
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        logger.info(f"Loading summary file: {summary_path}")
        
        # Read the summary file
        try:
            df = pd.read_excel(summary_path)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")
        
        # Validate required columns
        required_cols = ['Dataset Name', 'Split', 'Min', 'Max']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns in summary file: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Extract training split data
        training_data = df[df['Split'].str.lower() == 'training'].copy()
        
        if training_data.empty:
            raise KeyError("No 'Training' split found in summary file")
        
        logger.info(f"Found {len(training_data)} training datasets")
        
        # Calculate scaling factors (max - min) for each dataset
        scaling_factors = {}
        
        for _, row in training_data.iterrows():
            dataset_name = row['Dataset Name']
            min_val = row['Min']
            max_val = row['Max']
            
            if pd.isna(min_val) or pd.isna(max_val):
                logger.warning(f"Skipping dataset '{dataset_name}': Missing min/max values")
                continue
            
            range_val = max_val - min_val
            
            if range_val <= 0:
                logger.warning(
                    f"Invalid range for dataset '{dataset_name}': "
                    f"min={min_val}, max={max_val}, range={range_val}. "
                    f"Setting scaling factor to 1.0"
                )
                range_val = 1.0
            
            scaling_factors[dataset_name] = range_val
            logger.debug(f"{dataset_name}: range = {range_val:.4f} (min={min_val:.4f}, max={max_val:.4f})")
        
        if not scaling_factors:
            raise ValueError("No valid scaling factors could be extracted from summary file")
        
        logger.info(f"Successfully loaded {len(scaling_factors)} scaling factors")
        
        return cls(scaling_factors)
    
    def scale_interval(self, dataset_name: str, interval_width: float) -> float:
        """
        Scale an interval width by the dataset's training range.
        
        Scaled interval = interval_width / (max - min)
        
        This normalizes the interval to be relative to the data range,
        making it comparable across different scales.
        
        Args:
            dataset_name: Name of the dataset/model
            interval_width: Raw interval width to scale
            
        Returns:
            Scaled interval width (normalized by training range)
            
        Raises:
            KeyError: If dataset_name not found in scaling factors
        """
        if dataset_name not in self.scaling_factors:
            available = list(self.scaling_factors.keys())[:5]  # Show first 5
            raise KeyError(
                f"Dataset '{dataset_name}' not found in scaling factors.\n"
                f"Available datasets (showing first 5): {available}"
            )
        
        scaling_factor = self.scaling_factors[dataset_name]
        scaled = interval_width / scaling_factor
        
        return scaled
    
    def scale_dataframe(
        self, 
        df: pd.DataFrame, 
        dataset_col: str = 'Dataset Name',
        interval_col: str = 'Interval_weight',
        output_col: str = 'Scaled_Interval_weight'
    ) -> pd.DataFrame:
        """
        Scale interval widths in a DataFrame.
        
        Args:
            df: DataFrame containing interval widths
            dataset_col: Name of column containing dataset names
            interval_col: Name of column containing interval widths to scale
            output_col: Name of column for scaled intervals (added to df)
            
        Returns:
            DataFrame with added scaled interval column
            
        Raises:
            ValueError: If required columns are missing
        """
        if dataset_col not in df.columns:
            raise ValueError(f"Dataset column '{dataset_col}' not found in DataFrame")
        
        if interval_col not in df.columns:
            raise ValueError(f"Interval column '{interval_col}' not found in DataFrame")
        
        df = df.copy()
        
        # Scale each row
        scaled_intervals = []
        missing_datasets = set()
        
        for idx, row in df.iterrows():
            dataset = row[dataset_col]
            interval = row[interval_col]
            
            if pd.isna(interval):
                scaled_intervals.append(np.nan)
                continue
            
            try:
                scaled = self.scale_interval(dataset, interval)
                scaled_intervals.append(scaled)
            except KeyError:
                missing_datasets.add(dataset)
                scaled_intervals.append(np.nan)
        
        df[output_col] = scaled_intervals
        
        if missing_datasets:
            logger.warning(
                f"Could not scale intervals for {len(missing_datasets)} datasets: "
                f"{list(missing_datasets)[:5]}"  # Show first 5
            )
        
        n_scaled = df[output_col].notna().sum()
        logger.info(f"Successfully scaled {n_scaled}/{len(df)} intervals")
        
        return df
    
    def get_scaling_factor(self, dataset_name: str) -> Optional[float]:
        """
        Get the scaling factor for a dataset without raising an error.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Scaling factor (max - min) or None if not found
        """
        return self.scaling_factors.get(dataset_name)
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of scaling factors.
        
        Returns:
            DataFrame with dataset names and their scaling factors
        """
        data = [
            {'Dataset': name, 'Scaling_Factor': factor}
            for name, factor in sorted(self.scaling_factors.items())
        ]
        
        df = pd.DataFrame(data)
        
        logger.info("\nScaling Factor Statistics:")
        logger.info(f"  Count: {len(df)}")
        logger.info(f"  Mean:  {df['Scaling_Factor'].mean():.4f}")
        logger.info(f"  Std:   {df['Scaling_Factor'].std():.4f}")
        logger.info(f"  Min:   {df['Scaling_Factor'].min():.4f}")
        logger.info(f"  Max:   {df['Scaling_Factor'].max():.4f}")
        
        return df
    
    def save_scaling_factors(self, output_path: str):
        """
        Save scaling factors to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        df = self.get_summary_stats()
        df.to_csv(output_path, index=False)
        logger.info(f"Scaling factors saved to: {output_path}")


def load_and_scale_intervals(
    data_file: str,
    summary_file: str,
    dataset_col: str = 'Dataset Name',
    interval_col: str = 'Interval_weight',
    output_col: str = 'Scaled_Interval_weight'
) -> pd.DataFrame:
    """
    Convenience function to load data and scale intervals in one step.
    
    Args:
        data_file: Path to Excel/CSV file with interval data
        summary_file: Path to summary.xlsx with min/max values
        dataset_col: Name of dataset column
        interval_col: Name of interval column
        output_col: Name for scaled interval column
        
    Returns:
        DataFrame with scaled intervals
    """
    # Load scaler
    scaler = IntervalScaler.from_summary_file(summary_file)
    
    # Load data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)
    
    # Scale intervals
    df_scaled = scaler.scale_dataframe(df, dataset_col, interval_col, output_col)
    
    return df_scaled

