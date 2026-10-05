"""Evaluate regression predictions and save metrics, rows, and plots."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
	model,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	output_dir: str | Path,
	training_mean: float,
	test_rows: pd.DataFrame | None = None,
) -> dict[str, float | int]:
	"""Score held-out predictions against a training-mean baseline."""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	raw_predictions = np.asarray(model.predict(X_test), dtype=float)
	predictions = np.maximum(raw_predictions, 0)
	actual = np.asarray(y_test, dtype=float)
	baseline_predictions = np.full_like(actual, training_mean)
	metrics: dict[str, float | int] = {
		"mae": float(mean_absolute_error(actual, predictions)),
		"mse": float(mean_squared_error(actual, predictions)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
		"r2": float(r2_score(actual, predictions)),
		"baseline_mae": float(mean_absolute_error(actual, baseline_predictions)),
		"baseline_rmse": float(np.sqrt(mean_squared_error(actual, baseline_predictions))),
		"baseline_r2": float(r2_score(actual, baseline_predictions)),
		"negative_raw_prediction_count": int((raw_predictions < 0).sum()),
		"test_rows": int(len(actual)),
	}
	(output_path / "evaluation_metrics.json").write_text(
		json.dumps(metrics, indent=2), encoding="utf-8"
	)

	prediction_table = pd.DataFrame(
		{
			"actual_RUL_Cycles": actual,
			"predicted_RUL_Cycles": predictions,
			"raw_predicted_RUL_Cycles": raw_predictions,
		}
	)
	if test_rows is not None:
		for column in ("Equipment_ID", "Cycle"):
			if column in test_rows.columns:
				prediction_table.insert(0, column, test_rows[column].to_numpy())
	prediction_table.to_csv(output_path / "test_predictions.csv", index=False)

	figure, axis = plt.subplots(figsize=(7, 6))
	axis.scatter(actual, predictions, alpha=0.35, s=18)
	limits = [min(actual.min(), predictions.min()), max(actual.max(), predictions.max())]
	axis.plot(limits, limits, color="crimson", linestyle="--", label="Perfect prediction")
	axis.set(
		xlabel="Actual RUL (cycles)",
		ylabel="Predicted RUL (cycles)",
		title="Held-out predictions vs actual",
	)
	axis.legend()
	figure.tight_layout()
	figure.savefig(output_path / "actual_vs_predicted.png", dpi=140)
	plt.close(figure)

	residuals = actual - predictions
	figure, axis = plt.subplots(figsize=(7, 5))
	axis.scatter(predictions, residuals, alpha=0.35, s=18)
	axis.axhline(0, color="crimson", linestyle="--")
	axis.set(
		xlabel="Predicted RUL (cycles)",
		ylabel="Residual (actual - predicted)",
		title="Residuals on held-out equipment",
	)
	figure.tight_layout()
	figure.savefig(output_path / "residuals.png", dpi=140)
	plt.close(figure)
	return metrics