import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, HTML

"""
tasks/tutorial/load_dataset_classification.py
----------------------------------------------
Load and split a classification dataset for the tutorial pipeline.

Built-in datasets
-----------------
  tox21_sr_mmp : Tox21 SR-MMP assay (binary: active/inactive)
                 ~8k molecules; downloaded from MoleculeNet mirror.

  bbbp         : Blood-Brain Barrier Permeability (binary)
                 ~2k molecules; downloaded from MoleculeNet mirror.

  clintox      : ClinTox (binary: FDA approved / clinical trial toxicity)
                 ~1.5k molecules; downloaded from MoleculeNet mirror.

  csv / xlsx   : any local file with [smiles_col, target_col]

Outputs
-------
  product["train"]  : Excel Training sheet [Smiles, <target_col>]
  product["test"]   : Excel Test sheet     [Smiles, <target_col>]
  product["meta"]   : JSON metadata
"""

# + tags=["parameters"]
dataset      = "bbbp"
smiles_col   = "smiles"
target_col   = None
test_size    = 0.2
random_state = 42
product      = None
# -

# ── built-in loaders ──────────────────────────────────────────────────────────

def _bbbp() -> pd.DataFrame:
    """Blood-Brain Barrier Permeability (Martins et al. 2012) from DeepChem S3."""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    df = pd.read_csv(url)
    # columns: num, name, p_np, smiles
    df = df.rename(columns={"smiles": "Smiles", "p_np": "BBBP"})
    df = df[["Smiles", "BBBP"]].dropna()
    df["BBBP"] = df["BBBP"].astype(int).map({0: "non-permeable", 1: "permeable"})
    return df


def _clintox() -> pd.DataFrame:
    """ClinTox (Gayvert et al. 2016) from DeepChem S3."""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    df = pd.read_csv(url)
    # columns: smiles, FDA_APPROVED, CT_TOX
    df = df.rename(columns={"smiles": "Smiles"})
    df = df[["Smiles", "FDA_APPROVED"]].dropna()
    df["FDA_APPROVED"] = df["FDA_APPROVED"].astype(int).map(
        {0: "not_approved", 1: "approved"})
    return df


def _from_file(path, smiles_col_, target_col_):
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    df = df.rename(columns={smiles_col_: "Smiles"})
    return df[["Smiles", target_col_]].dropna()


# ── dispatch ──────────────────────────────────────────────────────────────────

BUILTIN = {
    "bbbp":    (_bbbp,    "BBBP"),
    "clintox": (_clintox, "FDA_APPROVED"),
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

class_counts = df[target_col].value_counts()
display(Markdown(f"## Dataset : {dataset}  |  target: {target_col}  |  n={len(df)}"))
display(Markdown("- Class distribution:"))
display(Markdown(class_counts.to_string()))

# Stratified split to preserve class balance
np.random.seed(random_state)
try:
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[target_col]
    )
except ValueError:
    # Fallback if stratification fails (very small classes)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

display(Markdown(f"### Train={len(train_df)}  Test={len(test_df)}"))

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
    "classes": sorted(df[target_col].unique().tolist()),
    "task": "classification",
}
with open(product["meta"], "w") as f:
    json.dump(meta, f, indent=2)

Markdown(f"- train → {product['train']}");Markdown(f"- test  → {product['test']}");Markdown(f"- meta  → {product['meta']}")
