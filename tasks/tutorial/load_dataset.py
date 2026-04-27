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
dataset = "esol"
smiles_col = "smiles"
target_col = None          # None → auto-detect from built-in datasets
test_size = 0.2
random_state = 42
product = None
# -

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, HTML


# ── built-in loaders ──────────────────────────────────────────────────────────

def _esol() -> pd.DataFrame:
    """ESOL (Delaney 2004) – aqueous solubility from DeepChem mirror."""
    url = (
        "https://raw.githubusercontent.com/deepchem/deepchem/refs/heads/master/datasets/delaney-processed.csv"
    )
    df = pd.read_csv(url)
    df = df.rename(columns={
        "smiles": "Smiles",
        "measured log solubility in mols per litre": "LogS",
    })
    return df[["Smiles", "LogS"]]


def _lipophilicity() -> pd.DataFrame:
    """Lipophilicity (AstraZeneca, Wu et al. 2018) from DeepChem S3 bucket."""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    df = pd.read_csv(url)
    # columns: CMPD_CHEMBLID, exp, smiles
    df = df.rename(columns={"smiles": "Smiles", "exp": "Lipophilicity"})
    return df[["Smiles", "Lipophilicity"]]


def _from_file(path: str, smiles_col_: str, target_col_: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df = df.rename(columns={smiles_col_: "Smiles", target_col_: target_col_})
    cols = ["Smiles", target_col_]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df


# ── dispatch ──────────────────────────────────────────────────────────────────

BUILTIN = {
    "esol": (_esol,  "LogS"),
    "lipo": (_lipophilicity, "Lipophilicity"),
}

if dataset in BUILTIN:
    loader_fn, default_target = BUILTIN[dataset]
    df = loader_fn()
    if target_col is None:
        target_col = default_target
    elif target_col != default_target and default_target in df.columns:
        df = df.rename(columns={default_target: target_col})
else:
    if target_col is None:
        raise ValueError(f"target_col must be specified for dataset='{dataset}'")
    df = _from_file(dataset, smiles_col, target_col)

df = df.dropna(subset=["Smiles", target_col])
df = df[df["Smiles"].astype(str).str.strip() != ""].reset_index(drop=True)

# ── split ─────────────────────────────────────────────────────────────────────
np.random.seed(random_state)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

Markdown(f"## Dataset : {dataset}  |  target: {target_col}");Markdown(f"Total={len(df)}  Train={len(train_df)}  Test={len(test_df)}")

# ── save ─────────────────────────────────────────────────────────────────────
Path(product["train"]).parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(product["train"], engine="xlsxwriter") as w:
    train_df.to_excel(w, sheet_name="Training", index=False)

with pd.ExcelWriter(product["test"], engine="xlsxwriter") as w:
    test_df.to_excel(w, sheet_name="Test", index=False)

meta = {
    "dataset": dataset,
    "target_col": target_col,
    "smiles_col": "Smiles",
    "n_total": len(df),
    "n_train": len(train_df),
    "n_test": len(test_df),
    "test_size": test_size,
    "random_state": random_state,
}
with open(product["meta"], "w") as f:
    json.dump(meta, f, indent=2)

Markdown(f"- train → {product['train']}");Markdown(f"- test  → {product['test']}");Markdown(f"- meta  → {product['meta']}")
