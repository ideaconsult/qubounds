# ad_comparison.py is no longer a standalone pipeline task.
#
# AD comparison logic has been moved into mapie_external.py, which reads
# pred_file, ad_cols, and ad_col_directions from dataset_config[dataset]
# in env.tutorial.yaml and runs the comparison as section S5 of the
# external predictions task.
#
# To enable AD comparison, add to your dataset entry in env.tutorial.yaml:
#
#   tpyriformis:
#     pred_file: "products/vega/TPYRIFORMIS/predictions.xlsx"
#     pred_col:  "Pred"
#     ad_cols:   ["ADI"]
#     ad_col_directions: ["similarity"]
#
# Then run:
#   ploomber build --entry-point pipeline.tutorial.yaml --env-file env.tutorial.yaml
