import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, HTML
from qubounds.vega.utils_vega import load_vega_report, writeExcel_epa, get_adi_cols


"""
tasks/tutorial/load_dataset.py
----------------------------
Download / prepare a regression dataset for the comparison pipeline.

Supported sources
-----------------
  csv   : any local CSV with columns [smiles_col, target_col]
  excel : any local .xlsx with columns [smiles_col, target_col]
  esol  : built-in ESOL dataset (Delaney 2004, aqueous solubility)
  lipo  : built-in Lipophilicity dataset (AstraZeneca)

Outputs
-------
  product["train"]  : Excel with Training sheet  [Smiles, <target_col>]
  product["test"]   : Excel with Test sheet       [Smiles, <target_col>]
  product["meta"]   : JSON with dataset metadata

Parameters (ploomber)
---------------------
  dataset        : str   – "esol", "lipo", or file path (.csv/.xlsx)
  smiles_col     : str   – SMILES column name (for file sources)
  target_col     : str   – target column name (required for file sources)
  test_size      : float – fraction for test split (default 0.2)
  random_state   : int   – seed (default 42)
  product        : dict  – ploomber product dict {train, test, meta}
"""

# + tags=["parameters"]
dataset_config = None
dataset_key = "esol"
test_size = 0.2
random_state = 42
product = None
task = None
# -


def _from_file(path: str, cfg: dict = {}) -> pd.DataFrame:
    pred_col          = cfg.get("pred_col",None)
    hard_col          = cfg.get("hard_col",None)    
    proba_col          = cfg.get("proba_col",None)
    target_col          = cfg.get("target_col", None)
    smiles_col    = cfg.get("smiles_col","Smiles")
    split_col         = cfg.get("split_col",None)
    split_train_value = cfg.get("split_train_value",   "TRAINING")
    split_test_value  = cfg.get("split_test_value",    "TEST")
    ad_cols           = cfg.get("ad_cols",[])
    ad_col_directions = cfg.get("ad_col_directions",   [])

    if not path.startswith("http"):
        p = Path(path)
    else:
        p = path

    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df = df.rename(columns={smiles_col: "Smiles"})
    cols = ["Smiles", target_col]
    if pred_col in df.columns:
        cols.extend([pred_col])
    if hard_col in df.columns:
        cols.extend([hard_col])        
    if proba_col in df.columns:
        cols.extend([proba_col])        
    for ad_col in ad_cols:        
        cols.extend([ad_col])
    if split_col is not None and split_col in df.columns:
        _vals = split_test_value if isinstance(split_test_value, list) else [split_test_value]
        _vals = [str(v).lower() for v in _vals]
        train_val = str(split_train_value).lower()
        # normalize once
        _col_lower = df[split_col].astype(str).str.lower()
        # assign only where explicitly matched
        df.loc[_col_lower == train_val, split_col] = "TRAINING"
        df.loc[_col_lower.isin(_vals), split_col] = "TEST"    
        df.rename(columns = {split_col: "Status"}, inplace=True)    
    else:
        # get indices for train/test
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state
        )
        df['Status'] = 'TRAINING'
        df.loc[test_idx, 'Status'] = 'TEST'
    cols.extend(["Status"])        
    return df[cols]


cfg               = dataset_config.get(dataset_key, {}) if isinstance(dataset_config, dict) else {}
dataset = cfg
dataset

pred_col          = cfg.get("pred_col",None)
hard_col          = cfg.get("hard_col",None)
prob_col          = cfg.get("prob_col",None)
target_col          = cfg.get("target_col", None)
target_descr          = cfg.get("target_descr", target_col)
smiles_col    = cfg.get("smiles_col","Smiles")
split_col         = cfg.get("split_col",None)
split_train_value = cfg.get("split_train_value",   "train")
split_test_value  = cfg.get("split_test_value",    "test")
ad_cols           = cfg.get("ad_cols",[])
ad_col_directions = cfg.get("ad_col_directions",   [])
n_quantile_bins   = int(cfg.get("n_quantile_bins", 5))

df = _from_file(dataset["path"], cfg)
df.shape

display(Markdown(f"## Dataset : {dataset_key}  |  target: {target_col}  |  n={len(df)}"))

if task == "classification":
    classes = {int(k): v for k, v in cfg.get("classes",{}).items()}    
    as_num = pd.to_numeric(df[target_col], errors="coerce")
    ratio_numeric = as_num.notna().mean()
    if ratio_numeric == 1:  # numeric classes
        valid = set(classes.keys())
    else:    
        valid = set(classes.values())
    print( df[target_col].unique(), valid)
    df[target_col] = df[target_col].where(df[target_col].isin(valid), np.nan)
    class_counts = df[target_col].value_counts(dropna=False)
    display(Markdown("- Class distribution:"))
    display(Markdown(class_counts.to_string()))   

print(df.columns)

df.head()

subset=["Smiles", target_col]
if hard_col in df.columns:
    subset.append(hard_col)
if prob_col in df.columns:
    subset.append(prob_col)    
df = df.dropna(subset=subset)
df = df[df["Smiles"].astype(str).str.strip() != ""].reset_index(drop=True)

# ── split ─────────────────────────────────────────────────────────────────────
np.random.seed(random_state)

model_json = { 
    "dataset": dataset_key,
    "results_name" : [target_descr],    
    "info": { "key": dataset_key,
            "name": dataset_key, 
            "version": "",
            "units": "",
            "Experimental" : target_col,
            "ADI": ad_cols,
            "ADI_direction": ad_col_directions,
    }, 
    "training_dataset": [dataset["path"]]}

if pred_col:
    model_json["results_name"] = [target_descr]
# ── save ─────────────────────────────────────────────────────────────────────
Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)

writeExcel_epa(
    product["data"],
    model_json,
    pred_value=pred_col if task == "regression" else hard_col,
    exp_value=target_col,
    df=df,
    adi_columns=ad_cols,
    software=cfg.get("software", None),
    extra_sheet=False
                    )


meta = {
    "dataset": dataset_key,
    "path": dataset["path"],
    "target_col": target_col,
    "split_col" : split_col,
    "split_train_value" : split_train_value,
    "split_test_value" : split_test_value,
    "ad_cols": ad_cols,
    "ad_col_directions": ad_col_directions,
    "smiles_col": "Smiles",
    "n_total": len(df),
    "test_size": test_size,
    "random_state": random_state,
    "task": task
}
if task == "classification":
    meta["classes"] =sorted(df[target_col].unique().tolist())
    meta["hard_col"] = hard_col
    meta["prob_col"] = prob_col    
else:
    meta["pred_col"] = pred_col    
with open(product["meta"], "w") as f:
    json.dump(meta, f, indent=2)

Markdown(f"- data → {product['data']}");Markdown(f"- meta  → {product['meta']}")
