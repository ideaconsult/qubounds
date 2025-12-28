import pandas as pd
import pickle


# + tags=["parameters"]
product = None
upstream = None
# -

combined_df = pd.DataFrame()
for key_star in upstream:
    for key in upstream[key_star]:
        model_path = upstream[key_star][key]["ncmodel"] 
        sigma_model = {}
        with open(model_path, "rb") as f:
            sigma_model = pickle.load(f)
        file_path = upstream[key_star][key]["data"]
        try:
            df = pd.read_excel(file_path, sheet_name="Metrics")
            if "Split" not in df.columns:
                df["Split"] = "Test"
            _key = key.replace("conformal_external_classification_", "").replace("conformal_vega_classification_", "").replace("mapiec_*","")
            df["Endpoint"] = _key
            meta = pd.read_excel(file_path, sheet_name="Summary sheet")
            df = pd.merge(meta, df, on=['Method Name', 'Split'], how='outer')        
            meta = pd.read_excel(file_path, sheet_name="Cover sheet", header=None, index_col=0).transpose()
            for t in ["Property Name", "Property Description", "Dataset Name", "Dataset Description", "Property Units", "nTraining", "nTEST","Min","Max"]:
                try:
                    df[t] = meta[t].iloc[0]
                except Exception:
                    df[t] = None
            for t in ['sigma_r2', 'sigma_rmse', 'sigma_mae']:
                df[t] = sigma_model.get(t, None)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception:
            pass

#combined_df['Relative Interval Width'] = combined_df['average_set_size'] / (combined_df['Max'] - combined_df['Min'])
combined_df.to_excel(product["data"], index=False)        


#combined_df.sort_values(by=['average_set_size'], ascending=True).head(25)