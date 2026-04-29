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
dataset_key = "bbbp"
test_size = 0.2
random_state = 42
product = None
dataset_config = None
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
classes = {int(k): v for k, v in dataset["classes"].items()}
smiles_col = dataset["smiles_col"]
target_col = dataset["target_col"]
print(dataset["classes"])
df = _from_file(dataset["path"], smiles_col, target_col)
try:
    df[target_col] = df[target_col].astype(int).map(classes)
except Exception:
    pass    
print(df.info())
df.head()


df = df.dropna(subset=["Smiles", target_col])
df = df[df["Smiles"].astype(str).str.strip() != ""].reset_index(drop=True)

class_counts = df[target_col].value_counts()
display(Markdown(f"## Dataset : {dataset_key}  |  target: {target_col}  |  n={len(df)}"))
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
test_df = test_df.reset_index(drop=True)

display(Markdown(f"### Train={len(train_df)}  Test={len(test_df)}"))

Path(product["train"]).parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(product["train"], engine="xlsxwriter") as w:
    train_df.to_excel(w, sheet_name="Training", index=False)

with pd.ExcelWriter(product["test"], engine="xlsxwriter") as w:
    test_df.to_excel(w, sheet_name="Test", index=False)

meta = {
    "dataset": dataset_key,
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
