import os.path
import pandas as pd


# + tags=["parameters"]
product = None
upstream = None
input_folder = None
inchi_cache = "cache_inchi.json"
target = "txt"
columns = None
field_id = None
# -


def excel_folder_to_single_txt(
    input_folder: str,
    output_file: str,
    columns_to_keep: list,
    dedupe_column: str,
    sheet_name: str | int = 0,  # always single sheet, default first
    dropna: bool = True
):
    """
    Merge all Excel files in a folder into a single output file,
    selecting specific columns, aggregating duplicates based on a key column,
    and retaining all distinct values from other columns.

    Parameters
    ----------
    input_folder : str
        Folder containing .xlsx/.xls files.
    output_file : str
        Path of the merged TXT file.
    columns_to_keep : list
        Columns to retain.
    dedupe_column : str
        Column used to detect duplicates.
    sheet_name : str | int
        Sheet name or index to load (default first sheet).
    dropna : bool
        Drop rows with missing values after subsetting.
    """

    all_dfs = []

    for file in os.listdir(input_folder):
        if file.lower().endswith((".xlsx", ".xls")):
            file_path = os.path.join(input_folder, file)
            # Always load a single sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(file_path, df.shape)
            # Keep only specified columns
            df = df[columns_to_keep]

            if dropna:
                df = df.dropna(subset=columns_to_keep)

            all_dfs.append(df)

    # Concatenate all files into a single DataFrame
    combined = pd.concat(all_dfs, ignore_index=True)

    # Aggregate duplicates: keep unique non-null values for each column
    def agg_unique(series):
        vals = sorted({str(v) for v in series.dropna().unique()})
        return " | ".join(vals)

    merged = combined.groupby(dedupe_column).agg(agg_unique).reset_index()

    # Write merged output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, sep="\t", index=False)

    print(f"Created merged file: {output_file}")


def excel_folder_to_json(
    input_folder: str,
    output_folder: str,
    columns_to_keep: dict,
    sheet_name: str | int | None = 0,
    dropna: bool = False
):
    """
    Load all Excel spreadsheets from a folder, select specified columns,
    optionally drop NA values, and write cleaned results to JSON.

    Parameters
    ----------
    input_folder : str
        Path to folder containing .xlsx or .xls files.
    output_folder : str
        Path to folder where CSVs will be written.
    columns_to_keep : dict
        Column names to retain / rename
    sheet_name : str | int | None
        Sheet name or index to load. Default is 0 (first sheet).
        Use None to load all sheets (they'll be concatenated).
    dropna : bool
        Drop rows with missing values after subsetting columns.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith((".xlsx", ".xls")):
            file_path = os.path.join(input_folder, file)
            
            # Load Excel file
            if sheet_name is None:
                # Load all sheets and concatenate
                df_dict = pd.read_excel(file_path, sheet_name=None)
                df = pd.concat(df_dict.values(), ignore_index=True)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Keep only specified columns
            df = df[columns_to_keep.keys()]

            df.rename(columns=columns_to_keep, inplace=True)
            cols_to_split = [col for col in df.columns if col.startswith("attr_")]
            for col in cols_to_split:
                df[col] = df[col].str.split("|")
            df["id"] = df[field_id]
            df["type_s"] = "chemical"
            #tbd - structure type guess polymer, UVCB, etc
            # Drop missing rows (optional)
            if dropna:
                df = df.dropna()

            # Build output filename
            base_name = os.path.splitext(file)[0]
            file_path = os.path.join(output_folder, f"{base_name}.json")

            # Write CSV
            df.to_json(file_path, orient="records", indent=2)

            print(f"Processed: {file} → {file_path}")
            #break


if target == "txt":
    excel_folder_to_single_txt(
        input_folder,
        output_file=product["data"],
        columns_to_keep=["DTXSID", "QSAR_READY_SMILES"],
        dedupe_column="QSAR_READY_SMILES")
else:
    excel_folder_to_json(
        input_folder,
        output_folder=product["data"],
        columns_to_keep=columns,
        dropna=False
    )