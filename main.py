"""Run EDA, clean data, train the regression model, and evaluate it."""

import argparse
from pathlib import Path

from evaluation.evaluation import evaluate_model
from src.eda import load_dataset, run_eda
from src.model_training import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "RUL_prediction_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Explore the RUL dataset, train linear regression, and evaluate it."
	)
	parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Input CSV path")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
	parser.add_argument("--target", default="RUL_Cycles", help="Regression target column")
	parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of equipment held out")
	args = parser.parse_args()

	dataset = load_dataset(args.data)
	eda_report = run_eda(dataset, args.output / "eda", args.target)
	model, X_test, y_test, test_rows, training_info = train_model(
		dataset,
		args.output / "model",
		target_column=args.target,
		test_size=args.test_size,
	)
	metrics = evaluate_model(
		model,
		X_test,
		y_test,
		args.output / "evaluation",
		training_mean=float(training_info["training_target_mean"]),
		test_rows=test_rows,
	)

	print(f"EDA report: {eda_report}")
	print(f"Trained model: {training_info['model_path']}")
	print(f"Training rows: {training_info['train_rows']:,}; test rows: {training_info['test_rows']:,}")
	print(f"Equipment held out: {training_info['test_equipment_count']}")
	print("Evaluation (nonnegative predictions):")
	for metric in ("mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2"):
		print(f"  {metric}: {metrics[metric]:.3f}")
	print(f"  negative raw predictions: {metrics['negative_raw_prediction_count']}")
	print(f"All outputs saved under: {args.output.resolve()}")


if __name__ == "__main__":
	main()