import json
import pandas as pd
import os.path


# + tags=["parameters"]
config = None
cache_path = None
product = None
input_key = None
enabled = False
# -


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
            df.rename(columns=columns_to_keep)

            # Drop missing rows (optional)
            if dropna:
                df = df.dropna()

            # Build output filename
            base_name = os.path.splitext(file)[0]
            file_path = os.path.join(output_folder, f"{base_name}.json")

            # Write CSV
            df.to_json(file_path)

            print(f"Processed: {file} → {file_path}")
            break


df_chem_path = config[input_key]["SMILES_INPUT"]

if enabled:
    excel_folder_to_json(df_chem_path)

#with open(product["json"], "w", encoding="utf-8") as f:
#    json.dump(cleaned_items, f)