"""Load, profile, and visualize the RUL regression dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_dataset(file_path: str | Path) -> pd.DataFrame:
	"""Load a CSV dataset and normalize whitespace in its column names."""
	path = Path(file_path).expanduser()
	if not path.is_file():
		raise FileNotFoundError(f"Dataset not found: {path}")
	dataset = pd.read_csv(path)
	dataset.columns = dataset.columns.astype(str).str.strip()
	return dataset


def run_eda(
	dataset: pd.DataFrame,
	output_dir: str | Path,
	target_column: str = "RUL_Cycles",
) -> Path:
	"""Write a data-quality/profile report and save exploratory plots."""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	data = dataset.copy()
	data.columns = data.columns.astype(str).str.strip()
	if target_column not in data.columns:
		raise ValueError(f"Target column '{target_column}' is not in the dataset.")

	numeric = data.select_dtypes(include=np.number)
	categorical_columns = data.select_dtypes(exclude=np.number).columns
	missing = data.isna().sum()
	duplicate_count = int(data.duplicated().sum())
	numeric_summary = numeric.describe().T

	lines = [
		"# Exploratory Data Analysis",
		"",
		f"- Rows: {len(data):,}",
		f"- Columns: {data.shape[1]:,}",
		f"- Exact duplicate rows: {duplicate_count:,}",
		f"- Numeric columns: {len(numeric.columns)}",
		f"- Categorical columns: {len(categorical_columns)}",
		"",
		"## Column types and missing values",
		"",
		"| Column | Type | Missing | Unique |",
		"|---|---|---:|---:|",
	]
	for column in data.columns:
		lines.append(
			f"| {column} | {data[column].dtype} | {int(missing[column])} | {data[column].nunique(dropna=True)} |"
		)

	lines.extend(["", "## Numeric summary", "", numeric_summary.to_string(float_format=lambda value: f"{value:.3f}"), ""])
	lines.extend(["## Potential outliers (IQR rule)", ""])
	lines.append("Counts are diagnostic only; valid sensor extremes are not automatically removed.")
	lines.extend(["", "| Column | Below Q1 - 1.5 IQR | Above Q3 + 1.5 IQR |", "|---|---:|---:|"])
	for column in numeric.columns:
		q1 = numeric[column].quantile(0.25)
		q3 = numeric[column].quantile(0.75)
		spread = q3 - q1
		below = int((numeric[column] < q1 - 1.5 * spread).sum())
		above = int((numeric[column] > q3 + 1.5 * spread).sum())
		lines.append(f"| {column} | {below} | {above} |")

	target = pd.to_numeric(data[target_column], errors="coerce").dropna()
	lines.extend(
		[
			"",
			"## Target review",
			"",
			f"- Target: `{target_column}`",
			f"- Valid target rows: {len(target):,}",
			f"- Target at minimum value 5: {int(target.eq(5).sum()):,} ({target.eq(5).mean():.1%})",
			"- Confirm whether the repeated value of 5 represents a capped minimum RUL; this affects interpretation near end-of-life.",
			"",
			"## Target correlations",
			"",
		]
	)
	if target_column in numeric.columns:
		correlations = numeric.corr(numeric_only=True)[target_column].drop(target_column).sort_values(
			key=lambda values: values.abs(), ascending=False
		)
		lines.append(correlations.to_string(float_format=lambda value: f"{value:.3f}"))
	else:
		lines.append("Target is not numeric and cannot be analyzed as a regression target.")

	report_path = output_path / "eda_report.md"
	report_path.write_text("\n".join(lines), encoding="utf-8")

	plot_columns = [column for column in numeric.columns if numeric[column].nunique() > 1]
	if plot_columns:
		columns_per_row = 4
		rows = int(np.ceil(len(plot_columns) / columns_per_row))
		figure, axes = plt.subplots(rows, columns_per_row, figsize=(16, 3.2 * rows))
		for axis, column in zip(np.asarray(axes).reshape(-1), plot_columns):
			sns.histplot(data[column].dropna(), kde=True, ax=axis)
			axis.set_title(column)
		for axis in np.asarray(axes).reshape(-1)[len(plot_columns):]:
			axis.remove()
		figure.tight_layout()
		figure.savefig(output_path / "numeric_distributions.png", dpi=140)
		plt.close(figure)

	if len(numeric.columns) > 1:
		figure, axis = plt.subplots(figsize=(14, 11))
		sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=axis)
		axis.set_title("Numeric feature correlation matrix")
		figure.tight_layout()
		figure.savefig(output_path / "correlation_matrix.png", dpi=140)
		plt.close(figure)

	figure, axis = plt.subplots(figsize=(9, 5))
	sns.histplot(target, bins=40, kde=True, ax=axis)
	axis.set(title=f"Distribution of {target_column}", xlabel=target_column)
	figure.tight_layout()
	figure.savefig(output_path / "target_distribution.png", dpi=140)
	plt.close(figure)
	return report_path
