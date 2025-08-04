import os.path
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import traceback


# + tags=["parameters"]
product = None
xml_path = None
reports = None
# -


def parse_model_xml(xml_string_or_path, reports="."):
    """
    Parse an XML file or string and extract model metadata.

    Parameters:
        xml_string_or_path (str): path to XML file or XML string
        reports (str): directory where report_<Key>.txt might exist

    Returns:
        pd.DataFrame with metadata including 'report' and 'has_report'
    """
    # Load XML
    if xml_string_or_path.strip().startswith("<"):
        root = ET.fromstring(xml_string_or_path)
    else:
        tree = ET.parse(xml_string_or_path)
        root = tree.getroot()

    version_elem = root.find("Version")
    vega_elem = root.find("Vega")

    # Extract fields
    name = version_elem.findtext("Name", default=None)
    key = version_elem.findtext("Key", default=None)
    summary = version_elem.findtext("Summary", default=None)

    version = None
    for child in version_elem.findall("Version"):
        version = child.text

    ts_path = vega_elem.findtext("TS", default=None)

    # Detect classification by presence of Class elements
    class_values = vega_elem.find("ClassValues")
    if class_values is not None and class_values.findall("Class"):
        model_type = "classification"
    else:
        model_type = "regression"

    # Report file info
    report_filename = f"report_{key}.txt" if key else None
    report_path = os.path.join(reports, report_filename) if report_filename else None
    has_report = os.path.exists(report_path) if report_path else False

    # Build DataFrame
    data = {
        "Key": [key],
        "Name": [name],
        "Summary": [summary],
        "Version": [version],
        "TS": [ts_path],
        "Type": [model_type],
        "report": [report_filename],
        "has_report": [has_report]
    }

    return pd.DataFrame(data)


def parse_all_model_xml_with_ts_line_count(pattern):
    """
    Parse all XML files in a directory and report number of lines in the corresponding .txt training set.

    Returns:
        DataFrame with columns: [Key, Name, Summary, Version, TS, TS_n_lines, __source_file__]
    """
    all_xml_files = glob.glob(pattern)

    dfs = []
    for xml_file in all_xml_files:
        try:
            df = parse_model_xml(xml_file, reports)
            df["__source_file__"] = os.path.basename(xml_file)

            # Get TS path and try to open the corresponding .txt
            ts_path = df.at[0, "TS"]

            if ts_path and ts_path.endswith(".dat"):
                txt_path = ts_path[:-4] + ".txt"
                # txt_path = txt_path.lstrip("/")
                txt_path = f"..{txt_path}"
                df["TS_txt"] = txt_path
                _path = os.path.join(xml_path, txt_path)
                print(_path)
                if os.path.exists(_path):
                    with open(_path, "r") as f:
                        n_lines = sum(1 for _ in f)
                    df["TS_n_lines"] = n_lines
                else:
                    df["TS_n_lines"] = None  # or 0 if you prefer
            else:
                df["TS_n_lines"] = None
            # df["report_exists"] = os.path.exists(os.path.join(reports, df["report"]))
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to parse {xml_file}: {e}")            
            # traceback.print_exc(e)


    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


df = parse_all_model_xml_with_ts_line_count(os.path.join(xml_path,  "*.xml"))
df.head()
df.to_excel(product["data"], index=False)