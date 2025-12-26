import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tasks.assessment.utils import init_logging
import pickle


# + tags=["parameters"]
product = None
upstream = None
# -

logger = init_logging(Path(product["nb"]).parent / "logs", "report.log")

combined_df = pd.DataFrame()
for key_star in upstream:
    for key in upstream[key_star]:
        model_path = upstream[key_star][key]["ncmodel"] 
        sigma_model = {}
        with open(model_path, "rb") as f:
            sigma_model = pickle.load(f)
            
        file_path = upstream[key_star][key]["data"]
        df = pd.read_excel(file_path, sheet_name="Metrics")
        logger.info(f"{key_star}\t{key}\t{file_path}")
        if "Split" not in df.columns:
            df["Split"] = "Test"
        _key = key.replace("conformal_external_regression_", "").replace("conformal_vega_regression_", "").replace("mapie_", "")
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

combined_df['Relative Interval Width'] = combined_df['Average Interval Width'] / (combined_df['Max'] - combined_df['Min'])
combined_df.to_excel(product["data"], index=False)        


#combined_df.sort_values(by=['Relative Interval Width'], ascending=True).head(25)

def plotsummary(df):
    # Create a color map for endpoints
    endpoints = df['Endpoint'].unique()
    colors = plt.cm.tab20.colors  # Up to 20 distinct colors
    color_map = {ep: colors[i % len(colors)] for i, ep in enumerate(endpoints)}

    # Scatter plot
    plt.figure(figsize=(8,6))
    for ep in endpoints:
        subset = df[df['Endpoint'] == ep]
        plt.scatter(
            subset['Empirical coverage'],
            subset['Relative Interval Width'],
            color=color_map[ep],
            alpha=0.8,
            s=60,
            label=ep
        )

    # Add nominal coverage line
    plt.axvline(x=0.9, color='red', linestyle='--', label='Nominal coverage 0.9')

    # Labels, title
    plt.xlabel('Empirical Coverage')
    plt.ylabel('Relative Interval Width')
    plt.title('Conformal Prediction: Coverage vs. Interval Width Across Endpoints')

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()


plotsummary(combined_df)