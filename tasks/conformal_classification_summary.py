import pandas as pd
import ast
import math
from collections import Counter


# + tags=["parameters"]
product = None
upstream = None
# -

combined_df = pd.DataFrame()
for key_star in upstream:
    for key in upstream[key_star]:
        file_path = upstream[key_star][key]["data"]
        df = pd.read_excel(file_path, sheet_name="Metrics")
        print(df.columns)
        print(df.head(1))
        df["Split"] = "Test"
        df["source"] = "VEGA" if "vega" in key_star else "EPA"   
        _key = key.replace("conformal_external_classification_", "").replace("conformal_vega_classification_", "")
        df["Endpoint"] = _key
        meta = pd.read_excel(file_path, sheet_name="Summary sheet")
        print(meta)
        df = pd.merge(meta, df, on=['Method Name', 'Split'], how='outer')        
        meta = pd.read_excel(file_path, sheet_name="Cover sheet", header=None, index_col=0).transpose()
        for t in ["Property Name", "Property Description", "Dataset Name", "Dataset Description", "Property Units", "nTraining", "nTEST","Min","Max"]:
            try:
                df[t] = meta[t].iloc[0]
            except Exception:
                df[t] = None

        combined_df = pd.concat([combined_df, df], ignore_index=True)

# combined_df['Relative Interval Width'] = combined_df['Average Interval Width'] / (combined_df['Max'] - combined_df['Min'])
combined_df.to_excel(product["data"], index=False)        


combined_df.sort_values(by=['average_set_size'], ascending=True).head(25)