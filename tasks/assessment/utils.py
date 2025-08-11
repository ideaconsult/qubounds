import pandas as pd


def prepare_properties(df_long, possible_props, id_cols=['No.', 'ID', 'Smiles']):
    print(df_long.info())
    # Unique chemicals (ensure we always return them even if no props are present)
    unique_chems = df_long[id_cols].drop_duplicates().reset_index(drop=True)

    existing_props = set(df_long['Property']).intersection(possible_props)

    # Filter only relevant rows
    df_filtered = df_long[df_long['Property'].isin(existing_props)]
    # Pivot to one row per chemical (if nothing to pivot, start from unique_chems)
    if df_filtered.empty:
        df_wide = unique_chems.copy()
    else:
        df_wide = df_filtered.pivot_table(
            index=id_cols,
            columns='Property',
            values='Property_mean',
            aggfunc='mean'
        ).reset_index()

        # if some chemicals in unique_chems were missing in pivot (shouldn't happen),
        # merge to ensure full set (safer)
        df_wide = unique_chems.merge(df_wide, on=['No.', 'ID', 'Smiles'], how='left')
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
