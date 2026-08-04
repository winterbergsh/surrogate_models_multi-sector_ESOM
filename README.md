# surrogate_models_multi-sector_ESOM

Static JavaScript dashboard that recreates core paper figures/tables for:

- XGB.py
- MLP.py
- GP.py
- XGB+classifier.py
- xgb_informed_classifier.py

The dashboard is designed for direct deployment on Netlify without a build step.

## Included data

The app reads bundled CSV files from ./data:

- regression_surrogate_run_log.csv
- classifier_2fold_performance.csv
- joint_learning_informed_classifier_run_log.csv

## Local preview

Open index.html directly, or serve the folder with any static server.

## Refresh data from source outputs

Use this PowerShell command from the repository root:

Set-Location "e:\Surrogate_model_test_runs\surrogate_models_multi-sector_ESOM"
Copy-Item "e:\Surrogate_model_test_runs\surrogate_test_sabine\Single_target\outputs\test regression methods\surrogate_run_log.csv" ".\data\regression_surrogate_run_log.csv" -Force
Copy-Item "e:\Surrogate_model_test_runs\surrogate_test_sabine\Single_target\outputs\effect adding classifier\2fold_performance.csv" ".\data\classifier_2fold_performance.csv" -Force
Copy-Item "e:\Surrogate_model_test_runs\surrogate_test_sabine\Single_target\outputs\joint learning models\joint_learning_informed_classifier_run_log.csv" ".\data\joint_learning_informed_classifier_run_log.csv" -Force

## Netlify

The repository includes netlify.toml with:

- publish directory: .

Deploy by connecting the repository in Netlify and leaving build command empty.

