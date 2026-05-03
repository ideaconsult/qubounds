import pandas as pd
import os.path
from pathlib import Path
from qubounds.vega.utils_vega import load_vega_report, writeExcel_epa, get_adi_cols
import re
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score


# + tags=["parameters"]
upstream = None
product = None
vega_models = None
vega_exported_sets = None
vega_reports = None
prefix = "report_"
# -


def take_first_predicted_col(model):
    print(model)
    return model in ["EW_TOXICITY", "BCF_MEYLAN"]


# pay attention to exp columns when coming from test set or report - they are not encoded/encoded!
def get_main_prediction_smthwrong(model, predicted_columns):
    main_predicted_column = None
    main_unit = None
    main_index = None
    for index, col in enumerate(predicted_columns):
        match = re.search(r"\[(.*?)\]", col)
        _unit = match.group(1) if match else None
        # generally, we take the first column
        if main_predicted_column is None:
            main_predicted_column = col
            main_unit = _unit
            main_index = index
        # but preferrence for units used in regulatory
        if not take_first_predicted_col(model):
            if _unit in ["mg/l", "mg/L", "mg/Kg", "ug/l", "days", "1/(mg/kg-day)", "1/(mg/kg-day)"]:
                main_predicted_column = col
                main_unit = _unit
                main_index = index
    return main_predicted_column, main_unit, main_index


def get_main_prediction(model, predicted_columns):
    main_predicted_column = predicted_columns[0]
    match = re.search(r"\[(.*?)\]", main_predicted_column)
    main_unit = match.group(1) if match else None
    return main_predicted_column, main_unit, 0


def prepare_reports(vega_list_models,  vega_exported_sets, vega_reports, product):
    for row in vega_list_models.itertuples(index=True):
        #if not row.has_report:
        #    continue
        model = row.Key
        classValues = row.ClassValues
        if row.ClassValues == "{}":
            model_type = "regression" 
            inverse_classes = None
        else:
            s = classValues.strip("{}")
            items = re.split(r',\s*(?=-?\d+\.?\d*=)', s)
            classes = {
                float(k.strip()): v.strip()
                for item in items
                for k, v in [item.split("=", 1)]
            }
            inverse_classes = {v: k for k, v in classes.items()}
            model_type = "classification"
        
        file_report = os.path.join(vega_reports, f"{prefix}{model}.txt")
        if not Path(file_report).exists():
            continue
        df, metadata = load_vega_report(file_report)
        df.columns = ['ID' if col.lower() == 'id' else col for col in df.columns]
        df = df[~df['SMILES'].astype(str).str.strip().str.upper().isin(
            ['SMILES', 'SMILES STRUCTURE', 'SMILES VEGA', 'VEGA SMILES',
             'SMI VEGA', 'MDL SMILES STRUCTURE', 'NEUTRALIZED (Kekulized)_VEGA SMILES'])]
        df.columns = ['Smiles' if col.lower() == 'smiles' else col for col in df.columns]

        # datasets exported using https://github.com/ideaconsult/quarkus-vega-cli/tree/main/vega-wrapper-app --export-dataset option
        # it has Status indicating Training/Test - this is not available in the reports produced by vega
        # Id	CAS	SMILES	Status	Experimental value 	Predicted value 
        df_ts = pd.read_csv(os.path.join(vega_exported_sets, f"{model}.txt"), sep="\t")
        df_ts.rename(columns={"Id" : "ID"}, inplace=True)

        df_ts['Status'] = df_ts['Status'].astype(str).str.upper()
        df = df[df["ID"].str.isdigit()].copy()
        df['ID'] = df['ID'].astype(str)
        df_ts['ID'] = df_ts['ID'].astype(str)
        assert df.shape[0] == df_ts.shape[0], '%s != %s' % (df.shape[0], df_ts.shape[0])
        start_idx = df.columns.get_loc('Assessment') + 1  # +1 to exclude 'Assessment'
        # end_idx = df.columns.get_loc('Experimental')      # Exclusive of 'Experimental'
        end_idx = next(
            (i for i, col in enumerate(df.columns) if col.strip().startswith("Experimental")),
            None  # fallback if not found
        )
        predicted_columns = df.columns[start_idx:end_idx]
        main_column, main_unit, main_index = get_main_prediction(model, predicted_columns)

        experimental_column = df.columns[end_idx]
        if main_index > 0:
            assert experimental_column.startswith("Experimental")
            match = re.search(r"\[(.*?)\]", experimental_column)
            _exp_unit = match.group(1) if match else None
            # they have mg/L and mg/l in the same file ...
            assert _exp_unit.lower() == main_unit.lower()

        status_index = df_ts.columns.get_loc('Status')
        last_col = df_ts.columns[status_index + 1]
        # otherwise we'll get thse renamed by the merge
        if last_col == "Experimental":
            last_col = "Observed"
            df_ts.rename(columns={'Experimental': last_col}, inplace=True)
        #df.rename(columns={'Experimental': experimental_column}, inplace=True)

        if 'CAS' not in df_ts.columns:
            df_ts['CAS'] = None        
        merged = pd.merge(df, df_ts[["ID", "Status", "CAS", last_col]], on='ID', how='left')
        #merged = pd.merge(df, df_ts[["ID", "Status", "CAS"]], on='ID', how='left')

        stats = {}
        if model_type == "classification":
            stats = {}
            for split in ["TRAINING", "TEST"]:
                mask = merged['Status'] == split
                if mask.any():
                    try:
                        accuracy = accuracy_score(
                            merged[main_column],
                            merged[last_col]
                        )
                        stats[f"Accuracy_{split}"] = accuracy
                        if accuracy < .5:
                            print(stats)
                    except Exception:
                        pass
        else:
             for split in ["TRAINING", "TEST"]:
                mask = merged['Status'] == split
                if not mask.any():
                    continue
                try:
                    df_sub = merged.loc[mask, [main_column, last_col]].copy()

                    # force numeric (invalid parsing -> NaN)
                    df_sub[main_column] = pd.to_numeric(df_sub[main_column], errors='coerce')
                    df_sub[last_col] = pd.to_numeric(df_sub[last_col], errors='coerce')
                    df_sub = df_sub.dropna()

                    if len(df_sub) == 0:
                        stats[f"R2_{split}"] = None
                        stats[f"RMSE_{split}"] = None
                        continue
                    r2 = r2_score(
                        df_sub.loc[mask, main_column],
                        df_sub.loc[mask, last_col]
                    )
                    rmse = root_mean_squared_error(
                        df_sub[main_column],
                        df_sub[last_col]
                    )                        
                    stats[f"R2_{split}"] = r2
                    stats[f"RMSE_{split}"] = rmse
                    if r2 < .5:
                        print(model, split, main_column, last_col, stats)
                except Exception as x:
                    print(x)
                    stats[f"R2_{split}"] = None
        # for compatibility with the pipeline we previously had this as encoded value 
        if inverse_classes is not None:
            merged[last_col] = merged[last_col].map(inverse_classes)

        model_json = { 
            "results_name": predicted_columns,
            "info": { "key": model,
                    "name": row.Name, 
                    "version": row.Version,
                    "units": main_unit,
                    "Experimental" : last_col,
                    "Stats": stats
                    }, 
            "training_dataset": []}
        _col_d = "Descriptors range check"
        if _col_d in merged.columns:
            merged[_col_d] = (
                merged[_col_d]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({'true': 1, 'false': 0})
            )
        writeExcel_epa(
            os.path.join(product[model_type], f"{model}.xlsx"),
            model_json,
            pred_value=main_column,
            exp_value=last_col,
            df=merged,
            adi_columns=get_adi_cols(),
            software="VEGA_NEW", extra_sheet=False,
            keep_empty=True
                         )


Path(product["regression"]).mkdir(parents=True, exist_ok=True)
Path(product["classification"]).mkdir(parents=True, exist_ok=True)

vega_list_models = pd.read_excel(vega_models)

prepare_reports(vega_list_models, vega_exported_sets, vega_reports, product)


