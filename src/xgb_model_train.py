"""Train and evaluate an XGBoost regressor for remaining useful life."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from evaluation.evaluation import evaluate_model
from src.eda import load_dataset
from src.model_training import clean_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "RUL_prediction_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "xgboost"


def _build_model(
	numeric_features: list[str],
	categorical_features: list[str],
	parameters: dict[str, int | float],
	random_state: int,
) -> Pipeline:
	transformers = []
	if numeric_features:
		transformers.append(
			("numeric", SimpleImputer(strategy="median"), numeric_features)
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
	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			(
				"regressor",
				XGBRegressor(
					**parameters,
					min_child_weight=10,
					subsample=0.9,
					colsample_bytree=0.9,
					reg_lambda=10.0,
					objective="reg:squarederror",
					eval_metric="rmse",
					tree_method="hist",
					n_jobs=-1,
					random_state=random_state,
				),
			),
		]
	)


def train_xgboost_model(
	dataset: pd.DataFrame,
	output_dir: str | Path = DEFAULT_OUTPUT,
	target_column: str = "RUL_Cycles",
	group_column: str = "Equipment_ID",
	test_size: float = 0.2,
	random_state: int = 42,
) -> dict[str, float | int]:
	"""Train XGBoost on one grouped split and save model, metadata, and evaluation."""
	data, cleaning_summary = clean_dataset(dataset, target_column, group_column)
	feature_data = data.drop(columns=[target_column, group_column])
	feature_data = feature_data.dropna(axis=1, how="all")
	if feature_data.shape[1] == 0:
		raise ValueError("No usable feature columns remain after excluding the target and equipment ID.")

	numeric_features = feature_data.select_dtypes(include="number").columns.tolist()
	categorical_features = feature_data.select_dtypes(exclude="number").columns.tolist()
	splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_indices, test_indices = next(
		splitter.split(feature_data, data[target_column], groups=data[group_column])
	)
	X_train = feature_data.iloc[train_indices]
	X_test = feature_data.iloc[test_indices]
	y_train = data[target_column].iloc[train_indices]
	y_test = data[target_column].iloc[test_indices]
	train_groups = data[group_column].iloc[train_indices]
	candidate_parameters: list[dict[str, int | float]] = [
		{"n_estimators": 100, "max_depth": 1, "learning_rate": 0.05},
		{"n_estimators": 250, "max_depth": 1, "learning_rate": 0.03},
		{"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05},
		{"n_estimators": 250, "max_depth": 2, "learning_rate": 0.03},
	]
	cv_splitter = GroupKFold(n_splits=min(3, train_groups.nunique()))
	cv_results: list[dict[str, object]] = []
	for parameters in candidate_parameters:
		fold_scores = []
		for fold_train, fold_validation in cv_splitter.split(
			X_train, y_train, groups=train_groups
		):
			candidate = _build_model(
				numeric_features, categorical_features, parameters, random_state
			)
			candidate.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
			predictions = np.maximum(candidate.predict(X_train.iloc[fold_validation]), 0)
			fold_scores.append(
				float(np.sqrt(mean_squared_error(y_train.iloc[fold_validation], predictions)))
			)
		cv_results.append(
			{
				"parameters": parameters,
				"mean_rmse": float(np.mean(fold_scores)),
				"std_rmse": float(np.std(fold_scores)),
			}
		)
	best_result = min(cv_results, key=lambda result: result["mean_rmse"])
	best_parameters = best_result["parameters"]
	model = _build_model(
		numeric_features, categorical_features, best_parameters, random_state
	)
	model.fit(X_train, y_train)

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	model_path = output_path / "xgboost_pipeline.joblib"
	joblib.dump(model, model_path)
	metadata: dict[str, object] = {
		"model": "XGBRegressor",
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
		"cv_method": "GroupKFold",
		"cv_folds": min(3, train_groups.nunique()),
		"cv_best_mean_rmse": best_result["mean_rmse"],
		"selected_parameters": best_parameters,
		"cv_results": cv_results,
		"test_size": test_size,
		"random_state": random_state,
		"model_path": str(model_path),
		**cleaning_summary,
	}
	(output_path / "training_metadata.json").write_text(
		json.dumps(metadata, indent=2), encoding="utf-8"
	)
	metrics = evaluate_model(
		model,
		X_test,
		y_test,
		output_path / "evaluation",
		training_mean=float(y_train.mean()),
		test_rows=data.iloc[test_indices].copy(),
	)
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description="Train and evaluate XGBoost for RUL regression.")
	parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Input CSV path")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Artifact directory")
	parser.add_argument("--target", default="RUL_Cycles", help="Regression target column")
	parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of equipment held out")
	args = parser.parse_args()

	metrics = train_xgboost_model(
		load_dataset(args.data),
		output_dir=args.output,
		target_column=args.target,
		test_size=args.test_size,
	)
	print(f"XGBoost artifacts saved under: {args.output.resolve()}")
	for metric in ("mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2"):
		print(f"  {metric}: {metrics[metric]:.3f}")
	print(f"  negative raw predictions: {metrics['negative_raw_prediction_count']}")


if __name__ == "__main__":
	main()