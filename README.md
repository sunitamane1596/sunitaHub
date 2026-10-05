# RUL Prediction

End-to-end exploratory analysis and linear regression for `RUL_Cycles` in
`data/RUL_prediction_dataset.csv`.

## Run

From this project directory, install the dependencies and run the complete
workflow:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Train and evaluate the separate XGBoost model. It selects shallow tree settings
with grouped cross-validation on training equipment, then evaluates once on the
same held-out equipment split used by the linear model:

```powershell
python -m src.xgb_model_train
```

To use a different dataset or output directory:

```powershell
python main.py --data path/to/data.csv --output path/to/output
```

The workflow profiles data types, missing values, duplicates, numeric summaries,
IQR outliers, correlations, and the target distribution. It removes exact
duplicate rows and rows with invalid target values. Feature missing values are
imputed within the training pipeline; numeric columns are standardized and
categorical columns one-hot encoded. Sensor outliers are reported, not removed
automatically, because extreme readings may be meaningful.
## Frontend dashboard

A browser-based dashboard is available to view the saved metrics, prediction
sample, and generated plots without using the terminal:

```powershell
python server.py
```

Then open `http://127.0.0.1:8000` in your browser.
The test split holds out complete `Equipment_ID` groups. The ID is excluded from
model features, reducing the chance that equipment-specific patterns leak into
the test score. This evaluates generalization to equipment not seen during
training. Predictions below zero are clipped to zero for evaluation and saved
predictions; the raw count is also reported.

## Outputs

- `outputs/eda/eda_report.md` and EDA plots
- `outputs/model/linear_regression_pipeline.joblib` and training metadata
- `outputs/evaluation/evaluation_metrics.json`, test predictions, and diagnostic plots
- `outputs/xgboost/` contains the XGBoost pipeline, its metadata, and separate evaluation results

The dataset has many target values exactly equal to 5. Confirm whether 5 is a
capped minimum RUL before interpreting estimates close to end-of-life.
