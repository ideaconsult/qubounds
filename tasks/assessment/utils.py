import pandas as pd
from typing import List, Tuple


def prepare_properties(df_long, possible_props, id_cols=['No.', 'ID', 'Smiles']):
    # Check ID columns exist
    missing = [c for c in id_cols if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing ID columns: {missing}")

    # Keep all molecules, even if no properties are found
    unique_chems = df_long[id_cols].drop_duplicates().reset_index(drop=True)

    # Keep only the properties we want that are actually present
    existing_props = [p for p in possible_props if p in df_long['Property'].unique()]
    if not existing_props:
        return unique_chems

    df_filtered = df_long[df_long['Property'].isin(existing_props)]

    # Pivot each metric separately
    mean_pivot = df_filtered.pivot(index=id_cols, columns='Property', values='Property_mean')
    lower_pivot = df_filtered.pivot(index=id_cols, columns='Property', values='Property_lower')
    upper_pivot = df_filtered.pivot(index=id_cols, columns='Property', values='Property_upper')

    # Rename bounds
    lower_pivot.columns = [f"{c}_lower" for c in lower_pivot.columns]
    upper_pivot.columns = [f"{c}_upper" for c in upper_pivot.columns]

    # Join them on the index, then bring back ID columns
    df_pivoted = mean_pivot.join([lower_pivot, upper_pivot]).reset_index()

    # Merge with unique_chems to ensure all molecules are included
    df_wide = unique_chems.merge(df_pivoted, on=id_cols, how='left')

    return df_wide


def run_in_chunks(df_long, chunk_size=10, estimate=None):
    """
    Split a long-format DataFrame into chunks of `chunk_size` unique compounds
    and run estimate on each chunk.

    Parameters
    ----------
    df_long : DataFrame
        Input in long format, must contain 'No.', 'ID', 'Smiles'
    chunk_size : int
        Number of unique compounds per chunk

    Returns
    -------
    DataFrame
        Concatenated results from all chunks
    """
    results = []

    # Get unique compound identifiers
    unique_chems = df_long[['No.', 'ID', 'Smiles']].drop_duplicates().reset_index(drop=True)

    # Create chunk ranges
    for start_idx in range(0, len(unique_chems), chunk_size):
        chem_chunk = unique_chems.iloc[start_idx:start_idx + chunk_size]

        # Filter original DF for these compounds
        df_chunk = df_long.merge(chem_chunk, on=['No.', 'ID', 'Smiles'])

        # Run the PBT/vPvB calculation
        chunk_result = estimate(df_chunk)

        results.append(chunk_result)

    # Combine all chunk results
    return pd.concat(results, ignore_index=True)


def fuzzy_memberships_from_interval(
    interval: Tuple[float, float], 
    thresholds: List[float]
) -> List[float]:
    """
    Calculate fuzzy membership scores of a property interval over threshold-defined classes.

    Args:
        interval: Tuple (L, U) - lower and upper bounds of the predicted interval.
        thresholds: Sorted list of thresholds [T1, T2, ..., Tn] dividing classes.

    Returns:
        memberships: List of floats representing membership in each class.
                     The length is len(thresholds) + 1.
    """
    L, U = interval
    if U < L:
        raise ValueError("Upper bound must be >= lower bound")

    # Define class intervals based on thresholds
    bounds = [float('-inf')] + thresholds + [float('inf')]
    interval_length = U - L

    memberships = []
    for i in range(len(bounds) - 1):
        low = bounds[i]
        high = bounds[i + 1]

        # Overlap between predicted interval and class interval
        overlap = max(0.0, min(U, high) - max(L, low))

        memberships.append(overlap)

    # Normalize memberships if interval length > 0
    if interval_length > 0:
        memberships = [m / interval_length for m in memberships]
    else:
        # If zero-length interval, assign membership fully to the class containing the point
        memberships = [0.0] * len(memberships)
        for i in range(len(bounds) - 1):
            if bounds[i] <= L < bounds[i + 1]:
                memberships[i] = 1.0
                break

    return memberships
