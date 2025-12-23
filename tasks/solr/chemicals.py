import json
import pandas as pd

# + tags=["parameters"]
config = None
cache_path = None
product = None
input_key = None
# -


df_chem_path = config[input_key]["SMILES_INPUT"]
df = pd.read_csv(df_chem_path, sep="\t")
df.head()

#with open(product["json"], "w", encoding="utf-8") as f:
#    json.dump(cleaned_items, f)