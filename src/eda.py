"""Load a CSV dataset and print a compact exploratory profile."""

import argparse
from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
	"""Load a CSV file into a pandas DataFrame."""
	path = Path(file_path).expanduser()
	if not path.is_file():
		raise FileNotFoundError(f"Dataset not found: {path}")
	return pd.read_csv(path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Load and inspect a CSV dataset.")
	default_dataset = Path(__file__).resolve().parent.parent / "data" / "RUL_prediction_dataset.csv"
	parser.add_argument(
		"file_path",
		nargs="?",
		type=Path,
		default=default_dataset,
		help="Path to the dataset CSV file (defaults to data/RUL_prediction_dataset.csv)",
	)
	args = parser.parse_args()

	dataset = load_dataset(args.file_path)
	print(f"Dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
	print("\nFirst five rows:")
	print(dataset.head())
	print("\nColumn types and non-null counts:")
	dataset.info()
	print("\nMissing values by column:")
	print(dataset.isna().sum())
	print("\nSummary statistics:")
	print(dataset.describe(include="all").T)


if __name__ == "__main__":
	main()
