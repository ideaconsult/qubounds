import pandas as pd
import os.path
from pathlib import Path
from tasks.vega.utils_vega import load_vega_report, writeExcel_epa, get_adi_cols
import re

# + tags=["parameters"]
upstream = None
product = None
ts_files = None
reports = None
# -


def take_first_predicted_col(model):
    print(model)
    return model in ["EW_TOXICITY", "BCF_MEYLAN"]


def get_main_prediction(model, predicted_columns):
    print("predicted_columns", predicted_columns)
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


def prepare_regression(vega_list_models, folder_path):
    for row in vega_list_models.itertuples(index=True):  # index=False to exclude index
        if not row.has_report:
            continue
        model = row.Key
        ts_file = os.path.join(folder_path, row.TS_txt)
        model_type = row.Type  # regr/class
        df, metadata = load_vega_report(os.path.join(folder_path, row.report))
        df.columns = ['ID' if col.lower() == 'id' else col for col in df.columns]
        df = df[~df['SMILES'].astype(str).str.strip().str.upper().isin(
            ['SMILES', 'SMILES STRUCTURE', 'SMILES VEGA', 'VEGA SMILES',
             'SMI VEGA', 'MDL SMILES STRUCTURE', 'NEUTRALIZED (Kekulized)_VEGA SMILES'])]
        df.columns = ['Smiles' if col.lower() == 'smiles' else col for col in df.columns]

        df_ts = pd.read_csv(ts_file, sep="\t")
        df_ts.columns = [col.strip() for col in df_ts.columns]
        replace_with_id = {'id', 'id algae noec', 'id algae ec50', 'no.', 
                           'mol_id', 'id fish lc50', 'no'}
        df_ts.columns = [
            "ID" if col.lower() in replace_with_id else col
            for col in df_ts.columns
        ]
        replace_with_status = {
            "set", "status", "split algae ec50", "split algae noec", "dataset", 
            "set orig. coral", "split fish lc50", "label", "class"
        }
        # Normalize column names by lowercasing and replacing if matched
        df_ts.columns = [
            "Status" if col.lower() in replace_with_status else col
            for col in df_ts.columns
        ]
        df_ts.columns = ['CAS' if col.lower() == 'cas' else col for col in df_ts.columns]
        print(model, df_ts.columns)
        print("df.columns", df.columns)

        df_ts['Status'] = df_ts['Status'].astype(str).str.upper()
        df = df[df["ID"].str.isdigit()].copy()
        df['ID'] = df['ID'].astype(str)
        df_ts['ID'] = df_ts['ID'].astype(str)
        assert df.shape[0] == df_ts.shape[0], '%s != %s' % (df.shape[0], df_ts.shape[0])
        print("All columns", df_ts.columns)
        start_idx = df.columns.get_loc('Assessment') + 1  # +1 to exclude 'Assessment'
        # end_idx = df.columns.get_loc('Experimental')      # Exclusive of 'Experimental'
        end_idx = next(
            (i for i, col in enumerate(df.columns) if col.strip().startswith("Experimental")),
            None  # fallback if not found
        )
        predicted_columns = df.columns[start_idx:end_idx]
        main_column, main_unit, main_index = get_main_prediction(model, predicted_columns)

        experimental_column = df.columns[end_idx]
        print("Experimental: ", experimental_column)
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
        #merged = pd.merge(df, df_ts[["ID", "Status", "CAS", last_col]], on='ID', how='left')
        merged = pd.merge(df, df_ts[["ID", "Status", "CAS"]], on='ID', how='left')

        print(model, merged.columns)
        model_json = { 
            "results_name": predicted_columns,
            "info": { "key": model,
                    "name": row.Name, 
                    "version": row.Version,
                    "units": main_unit,
                    "Experimental" : experimental_column,
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
            exp_value=experimental_column,
            df=merged,
            adi_columns=get_adi_cols(),
            software="VEGA_NEW"
                         )


Path(product["regression"]).mkdir(parents=True, exist_ok=True)
Path(product["classification"]).mkdir(parents=True, exist_ok=True)

vega_list_models = pd.read_excel(upstream["vega_list_models"]["data"])
print(vega_list_models.shape)
#_tmp = vega_list_models.loc[vega_list_models["Key"] == "FISH_KNN"]
#_tmp

prepare_regression(vega_list_models, folder_path=Path(reports))
#prepare_regression(_tmp, folder_path=Path(reports))

