"""Clean data, split by equipment, and train a linear regression pipeline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataset(
	dataset: pd.DataFrame,
	target_column: str = "RUL_Cycles",
	group_column: str = "Equipment_ID",
) -> tuple[pd.DataFrame, dict[str, int]]:
	"""Normalize columns, remove exact duplicates, and discard invalid targets."""
	data = dataset.copy()
	data.columns = data.columns.astype(str).str.strip()
	if data.columns.duplicated().any():
		duplicates = data.columns[data.columns.duplicated()].tolist()
		raise ValueError(f"Duplicate column names after trimming whitespace: {duplicates}")
	if target_column not in data.columns:
		raise ValueError(f"Target column '{target_column}' is not in the dataset.")
	if group_column not in data.columns:
		raise ValueError(f"Group column '{group_column}' is required for an equipment-level split.")

	original_rows = len(data)
	data = data.drop_duplicates().copy()
	duplicate_rows_removed = original_rows - len(data)
	data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
	invalid_target_rows = int(data[target_column].isna().sum())
	data = data.dropna(subset=[target_column]).reset_index(drop=True)
	if data[group_column].isna().any():
		raise ValueError(f"'{group_column}' contains missing equipment IDs; cannot make a grouped split.")
	return data, {
		"duplicate_rows_removed": duplicate_rows_removed,
		"invalid_target_rows_removed": invalid_target_rows,
		"rows_after_cleaning": len(data),
	}


def train_model(
	dataset: pd.DataFrame,
	output_dir: str | Path,
	target_column: str = "RUL_Cycles",
	group_column: str = "Equipment_ID",
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, dict[str, object]]:
	"""Fit linear regression and return held-out rows for evaluation."""
	data, cleaning_summary = clean_dataset(dataset, target_column, group_column)
	feature_data = data.drop(columns=[target_column, group_column])
	feature_data = feature_data.dropna(axis=1, how="all")
	if feature_data.shape[1] == 0:
		raise ValueError("No usable feature columns remain after excluding the target and equipment ID.")

	numeric_features = feature_data.select_dtypes(include="number").columns.tolist()
	categorical_features = feature_data.select_dtypes(exclude="number").columns.tolist()
	transformers = []
	if numeric_features:
		transformers.append(
			(
				"numeric",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				numeric_features,
			)
		)
	if categorical_features:
		transformers.append(
			(
				"categorical",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("encoder", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				categorical_features,
			)
		)
	preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
	model = Pipeline(
		steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
	)

	splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_indices, test_indices = next(
		splitter.split(feature_data, data[target_column], groups=data[group_column])
	)
	X_train = feature_data.iloc[train_indices]
	X_test = feature_data.iloc[test_indices]
	y_train = data[target_column].iloc[train_indices]
	y_test = data[target_column].iloc[test_indices]
	model.fit(X_train, y_train)

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	model_path = output_path / "linear_regression_pipeline.joblib"
	joblib.dump(model, model_path)
	metadata: dict[str, object] = {
		"target_column": target_column,
		"group_column_excluded_from_features": group_column,
		"feature_columns": feature_data.columns.tolist(),
		"numeric_features": numeric_features,
		"categorical_features": categorical_features,
		"train_rows": len(train_indices),
		"test_rows": len(test_indices),
		"train_equipment_count": int(data.iloc[train_indices][group_column].nunique()),
		"test_equipment_count": int(data.iloc[test_indices][group_column].nunique()),
		"training_target_mean": float(y_train.mean()),
		"test_size": test_size,
		"random_state": random_state,
		"model_path": str(model_path),
		**cleaning_summary,
	}
	(output_path / "training_metadata.json").write_text(
		json.dumps(metadata, indent=2), encoding="utf-8"
	)
	test_rows = data.iloc[test_indices].copy()
	return model, X_test, y_test, test_rows, metadata
