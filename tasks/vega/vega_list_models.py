import os.path
from qubounds.vega.utils_vega import parse_all_model_xml_with_ts_line_count


# + tags=["parameters"]
product = None
xml_path = None
reports = None
# -

pattern, xml_path, report
df = parse_all_model_xml_with_ts_line_count(os.path.join(xml_path,  "*.xml"))
df.head()
df.to_excel(product["data"], index=False)