import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, HTML


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
# -


def _from_file(path: str, smiles_col_: str, target_col_: str) -> pd.DataFrame:
    if not path.startswith("http"):
        p = Path(path)
    else:
        p = path

    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df = df.rename(columns={smiles_col_: "Smiles", target_col_: target_col_})
    cols = ["Smiles", target_col_]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df


dataset = dataset_config.get(dataset_key, None)
smiles_col = dataset["smiles_col"]
target_col = dataset["target_col"]
print(dataset)
df = _from_file(dataset["path"], smiles_col, target_col)
df.head()


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
    "dataset": dataset_key,
    "path": dataset["path"],
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
