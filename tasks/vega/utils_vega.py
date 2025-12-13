import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from rdkit import Chem
import os.path
import json
import plotly.express as px
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import chardet
import ast
from tasks.assessment.thresholds import Thresholds
import re


def load_vega_models(file_vega_models, model):
    df_models = pd.read_excel(file_vega_models, engine="openpyxl")
    df_models = df_models.loc[df_models["Key"] == model]

    needs_fix = df_models['ClassValues'].astype(str).str.contains('Âµ').any()
    if needs_fix:
        print("Column ClassValues contains garbled encoding — fixing...")
        df_models['ClassValues'] = df_models['ClassValues'].str.replace('Âµ', 'µ', regex=False)
    classvalues_str = df_models.loc[df_models["Key"] == model, "ClassValues"].values[0]
    classvalues_dict = parse_classvalues(classvalues_str)
    return df_models, classvalues_str, classvalues_dict


def load_vega_report(file):
    print("load_vega_report")
    with open(file, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']    
    print(f'{file} encoding {encoding}')
    with open(file, 'r', encoding=encoding, errors='ignore') as f:
        lines = [next(f).strip() for _ in range(4)]  # Read the first 4 lines

    # Extract metadata
    metadata = {
        'title': lines[0],
        'model_name': lines[1],
        'model_version': lines[2]
    }    
    return pd.read_csv(file, sep="\t", skiprows=4,  encoding=encoding, encoding_errors="ignore"), metadata


def clean_vega_report_df(df):
    df.columns = ['ID' if col.lower() == 'id' else col for col in df.columns]
    has_string = df['ID'].astype(str).str.strip().ne('').any()

    df = df[~df['SMILES'].astype(str).str.strip().str.upper().isin(
        ['SMILES', 'SMILES STRUCTURE', 'SMILES VEGA', 'VEGA SMILES',
            'SMI VEGA', 'MDL SMILES STRUCTURE', 'NEUTRALIZED (Kekulized)_VEGA SMILES'])]
    df.columns = ['Smiles' if col.lower() == 'smiles' else col for col in df.columns]
    if not has_string:
        df = df[df["ID"].str.isdigit()].copy()
    df['ID'] = df['ID'].astype(str)
    start_idx = df.columns.get_loc('Assessment') + 1  # +1 to exclude 'Assessment'
    end_idx = next(  # Exclusive of 'Experimental'
        (i for i, col in enumerate(df.columns) if col.strip().startswith("Experimental")),
        None  # fallback if not found
    )
    predicted_columns = df.columns[start_idx:end_idx]
    _col_d = "Descriptors range check"
    if _col_d in df.columns:
        df[_col_d] = (
            df[_col_d]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({'true': 1, 'false': 0})
        )
    return df, predicted_columns

def plot_histogram(df, column, pngfile):
    # Plot histogram
    plt.figure(figsize=(12, 6))
    df[column].hist(bins=50, color='skyblue', edgecolor='gray')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values')

    # Save as PNG
    plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory if you're doing this in a loop


def plot_histogram_grouped(df, value_column, group_column, title, pngfile):
    """
    Plots grouped histograms of a numeric column, color-coded by a categorical group.
    """
    plt.figure(figsize=(12, 6))

    # Generate distinct colors for each group
    groups = sorted(df[group_column].dropna().unique())
    n_colors = len(groups)
    cmap = plt.get_cmap('tab10' if n_colors <= 10 else 'tab20')
    colors = [cmap(i % cmap.N) for i in range(n_colors)]

    # Plot each group
    for color, group_name in zip(colors, groups):
        group_values = df[df[group_column] == group_name][value_column].dropna()
        plt.hist(group_values,
                 bins=50,
                 alpha=0.3,
                 label=str(group_name),
                 color=color,
                 density=True,
                 edgecolor='black',
                 histtype='stepfilled')

    # Annotate
    plt.xlabel(value_column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of "{title}" grouped by "{group_column}"')
    plt.legend(title=group_column)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Make x-axis more readable
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit number of ticks
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))  # Format floats to 2 decimals

    # Save figure
    plt.tight_layout()
    plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_violin_grouped(df, value_column, group_column, title, pngfile):
    """
    Plots violin plots of a numeric column grouped by a categorical group using matplotlib only.
    """
    # Drop rows with missing data
    df_clean = df.dropna(subset=[value_column, group_column])

    # Prepare data: group by group_column
    grouped = df_clean.groupby(group_column)[value_column]
    group_labels = []
    data = []

    for name, group in grouped:
        group_labels.append(name)
        data.append(group.values)

    # Set up figure
    plt.figure(figsize=(12, 6))

    # Create violin plot
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

    # Customize violins (optional)
    for pc in parts['bodies']:
        pc.set_facecolor('#9999ff')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    # Style medians
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

    # Set x-ticks
    plt.xticks(ticks=range(1, len(group_labels)+1), labels=group_labels, rotation=45, ha='right')
    plt.xlabel(group_column)
    plt.ylabel(value_column)

    # Format y-axis
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.title(f'Violin plot of "{title}" grouped by "{group_column}"')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save figure
    if pngfile is None:
        plt.show()
    else:    
        plt.savefig(pngfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_kde_plotly_simple(df, value_column, group_column, htmlfile, title=""):
    """
    Creates an interactive KDE plot grouped by a categorical column, and saves to an HTML file.

    Parameters:
        df (pd.DataFrame): Input data.
        value_column (str): Numeric column for density estimation.
        group_column (str): Categorical column to group KDEs.
        htmlfile (str): Output path for interactive HTML file.
        title (str): Optional plot title.
    """

    _title = f'"{title}" grouped by "{group_column}"'

    fig = px.density_estimate(
        df, x=value_column, color=group_column,
        title=_title
    )
    fig.update_layout(template='plotly_white')
    fig.write_html(htmlfile)


def generate_inchikeys(df, smiles_column='SMILES', cache_file='inchikey_cache.json'):
    """
    Convert a DataFrame with SMILES strings to InChIKeys using a persistent JSON cache.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a column of SMILES strings.
        smiles_column (str): Name of the column containing SMILES strings.
        cache_file (str): Path to JSON file for storing/retrieving cached InChIKeys.

    Returns:
        pd.DataFrame: Original DataFrame with a new 'InChIKey' column.
    """
    # Load cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            inchikey_cache = json.load(f)
    else:
        inchikey_cache = {}

    # Internal conversion function
    def smiles_to_inchikey(smiles):
        if smiles in inchikey_cache:
            return inchikey_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        inchikey = Chem.MolToInchiKey(mol) if mol else None
        inchikey_cache[smiles] = inchikey
        return inchikey

    # Apply function to column
    df['InChIKey'] = df[smiles_column].apply(smiles_to_inchikey)

    # Save updated cache
    with open(cache_file, 'w') as f:
        json.dump(inchikey_cache, f, indent=2)

    return df


def pairwise_comparisons_tukey(df, group_col='coating_system', value_col='value'):
    # Drop NA in the value column
    df_clean = df.dropna(subset=[value_col, group_col])
    
    # Perform Tukey HSD test
    tukey_result = pairwise_tukeyhsd(endog=df_clean[value_col],
                                     groups=df_clean[group_col],
                                     alpha=0.05)
    
    # Print summary (optional)
    print(tukey_result.summary())
    
    # Plot confidence intervals (optional)
    tukey_result.plot_simultaneous()
    plt.title(f"Tukey HSD pairwise comparisons for {value_col}")
    plt.show()
    
    # Convert summary to pandas DataFrame
    summary_data = tukey_result.summary().data
    # summary_data is a list of lists: first row is header, remaining are data rows
    df_results = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['meandiff', 'p-adj', 'lower', 'upper']
    for col in numeric_cols:
        df_results[col] = pd.to_numeric(df_results[col])
    
    return df_results


def classify_bcf(value, scale='bcf'):
    """
    Classify bioaccumulation potential based on BCF or logBCF value.

    Parameters:
        value (float): BCF or log10(BCF) value.
        scale (str): 'bcf' (default) if value is BCF,
                     'logbcf' if value is log10(BCF).

    Returns:
        str: Classification label:
             - 'Not bioaccumulative'
             - 'Bioaccumulative (B)'
             - 'Very bioaccumulative (vB)'
             - 'Invalid input'
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return 'Invalid input'

    if scale.lower() == 'logbcf':
        # Convert logBCF to BCF
        if v < 0:
            return 'Invalid input'
        bcf = 10 ** v
    elif scale.lower() == 'bcf':
        if v < 0:
            return 'Invalid input'
        bcf = v
    else:
        return 'Invalid input: scale must be "bcf" or "logbcf"'

    if bcf < 2000:
        return 'Not bioaccumulative'
    elif 2000 <= bcf < 5000:
        return 'Bioaccumulative (B)'
    else:
        return 'Very bioaccumulative (vB)'


def classify_lc50_fathead_minnow(lc50_value):
    """
    Classify toxicity based on LC50 (mg/L) for Fathead Minnow.

    Parameters:
        lc50_value (float): LC50 concentration in mg/L

    Returns:
        str: Toxicity classification
    """
    try:
        lc50 = float(lc50_value)
    except (ValueError, TypeError):
        return 'Invalid LC50'

    if lc50 <= 0:
        return 'Invalid LC50'
    elif lc50 <= 1:
        return 'Very toxic'
    elif lc50 <= 10:
        return 'Toxic'
    elif lc50 <= 100:
        return 'Harmful'
    else:
        return 'Not harmful / Low toxicity'


def get_props():
    return ["Predicted LogP", "Predicted BCF [log(L/kg)]", "Predicted toxicity [-log(mol/l)]",
                 "Predicted Henry's law [log atm-m3/mole]", "Predicted HL [log(days)]",
                 "Predicted EC50 [log(mg/L)]", "Predicted daphnia EC50 (log form) [log(mmol/l))]",
                 "Predicted LC50 [-log(mmol/L)]","Liver LOAEL [log(mg/kg bw)]", "Melting Point [°C]"]  


def writeExcel_epa(output_file, model_json, 
                   pred_value="PredictedValue", exp_value="ExperimentalValue",
                   df=None, adi_columns=None, software="VEGA"):
    key = model_json.get("info",{}).get("key", None)
    name = model_json.get("info",{}).get("name", None)
    version = model_json.get("info",{}).get("version", None)
    if key is None:
        return
    
    if df is None:
        df = pd.DataFrame(model_json["training_dataset"])
    if "Status" in df.columns:
        training_df = df.loc[df["Status"] == "TRAINING"]
        test_df = df.loc[df["Status"] == "TEST"]
        if test_df.empty:
            test_df = training_df
    else:
        training_df = df
        test_df = pd.DataFrame()

    stats = model_json["info"].get("Stats",{})
    metadata = [
        ("Property Name", pred_value),
        ("Property Description",  "; ".join(model_json["results_name"])),
        ("Dataset Name", key),
        ("Dataset Description", name),
        ("Property Units",  model_json["info"].get("units",None)),
        ("Experimental",  exp_value),
        ("nTraining",  stats.get("n_Train", None if training_df is None else training_df.shape[0])),
        ("nTEST", stats.get("n_Test", None if test_df is None else test_df.shape[0]))
        #  ("Model version", version),
    ]
        
    if exp_value in df.columns:
        numeric_values = pd.to_numeric(df[exp_value], errors='coerce')
        exp_min = numeric_values.min()
        exp_max = numeric_values.max()
    else:
        exp_min = None
        exp_max = None

    cover_df = pd.DataFrame(metadata)
    new_rows = pd.DataFrame([["Min", exp_min], ["Max", exp_max]],
                            columns=cover_df.columns)
    cover_df = pd.concat([cover_df, new_rows], ignore_index=True)


    summary = []
    summary.append({"Split": "Training",
                    "R2": stats.get("R2_Train", None),
                    "RMSE": stats.get("RMSE_Train", None),
                    "Accuracy": stats.get("Accuracy_Train", None),
                    "Sensitivity": stats.get("Sensitivity_Train", None),
                    "Specificity": stats.get("Specificity_Train", None),
                    })
    summary.append({"Split": "Test", 
                    "R2": stats.get("R2_Test", None),
                    "RMSE": stats.get("RMSE_Test", None),
                    "Accuracy": stats.get("Accuracy_Train", None),
                    "Sensitivity": stats.get("Sensitivity_Test", None),
                    "Specificity": stats.get("Specificity_Train", None),
                    }
                    ) 
    for row in summary:
        row["Dataset Name"] = key
        row["Descriptor Software"] = software
        row["Method Name"] = key
    summary_df = pd.DataFrame(summary, columns=[
        "Dataset Name", "Descriptor Software", "Method Name", "Split", "R2", "RMSE","Accuracy", "Sensitivity", "Specificity"])
    summary_df["Dataset Name"] = key
    summary_df["Descriptor Software"] = software
    summary_df["Method Name"] = key
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        cover_df.to_excel(writer, sheet_name='Cover sheet', index=False, header=False)
        summary_df.to_excel(writer, sheet_name='Summary sheet', index=False)
        training_df.to_excel(writer, sheet_name='Training', index=False)
        test_df.to_excel(writer, sheet_name='Test', index=False)
        try:
            cols = ["Smiles", exp_value, pred_value]
            if adi_columns is not None:
                for col in adi_columns:
                    if col in test_df.columns:
                        cols.append(col)
                print(cols)
            df = test_df[cols].rename(columns={
                pred_value: key,
                exp_value: "Exp"
            })
            df.to_excel(writer, sheet_name=key, index=False)
        except Exception as x:
            print(key, x)

    print(f"Excel file '{output_file}' created with all sheets.")


def get_adi_cols():
    return ["Similarity index", "Accuracy index",
                         "Predicted LogP (Meylan/Kowwin)", "MW", "MolecularWeight"]
    #    return ["Similarity index","Accuracy index",
    #                     "Concordance index",
    #                     "Max error index",
    #                     "Descriptors range check", "ACF index",
    #                     "Predicted LogP (Meylan/Kowwin)", "MW"]    


def parse_classvalues(classvalues_str):
    """
    Convert a string like '{0.0=NON-Toxic, 1.0=Toxic, -1.0=Not predicted}'
    into a Python dictionary: {0.0: 'NON-Toxic', 1.0: 'Toxic', -1.0: 'Not predicted'}

    {0.0=NON-Toxicant, 1.0=Toxicant, -1.0=Not Classifiable, 2.0=Reproductive and developmental toxicant, 
    4.0=Reproductive toxicant (no data on developmental toxicity), 
    8.0=Developmental toxicant (no data on reproductive toxicity), 
    9.0=Developmental NON-toxicant (no data on reproductive toxicity), 
    5.0=Developmental toxicant, reproductive NON-toxicant, 
    10.0=No data on reproductive and developmental toxicity, 
    3.0=Reproductive toxicant, developmental NON-toxicant, 
    6.0=Reproductive and developmental NON-toxicant, 
    7.0=Reproductive NON-toxicant (no data on developmental toxicity)}
    """

    for patch in ["5.0=Developmental toxicant, reproductive NON-toxicant","3.0=Reproductive toxicant, developmental NON-toxicant"]:
        classvalues_str = classvalues_str.replace(patch, patch.replace(",","_"))

    # Remove surrounding braces
    content = classvalues_str.strip('{}')

    # Split into key=value pairs
    pairs = content.split(',')

    corrected_pairs = []
    for pair in pairs:
        key, val = pair.split('=')
        key = key.strip()
        val = val.strip()
        # Quote the value properly to make it a valid Python string
        val_quoted = repr(val)
        corrected_pairs.append(f"{key}: {val_quoted}")

    # Rebuild the string with ':' instead of '=' and quoted values
    corrected_str = "{" + ", ".join(corrected_pairs) + "}"

    # Safely evaluate the string into a dictionary
    return ast.literal_eval(corrected_str)


def replace_labels_with_keys(df, column, classvalues_dict):
    """
    Replace text labels in df[column] with numeric keys from classvalues_dict.

    Parameters:
    - df: pandas DataFrame
    - column: str, column name with text labels
    - classvalues_dict: dict mapping numeric keys to text labels

    Returns:
    - DataFrame with new column '{column}_numeric' with replaced numeric values
    """
    # invert dict to map text label -> numeric key
    inverse_dict = {v: k for k, v in classvalues_dict.items()}

    df[f'{column}_numeric'] = df[column].map(inverse_dict)

    return df, f'{column}_numeric'


def get_main_prediction(model, predicted_columns):
    print("predicted_columns", predicted_columns)
    main_predicted_column = predicted_columns[0]
    match = re.search(r"\[(.*?)\]", main_predicted_column)
    main_unit = match.group(1) if match else None
    return main_predicted_column, main_unit, 0