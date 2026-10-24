
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.emi import emi_impute


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MISSINGNESS_DIR = PROJECT_DIR / "data" / "missingness"
MANIFEST_PATH = MISSINGNESS_DIR / "manifest.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "emi" / "pilot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    """Find a manifest column using supported column names."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not identify the {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def resolve_path(value):
    """Resolve a path stored in the manifest."""
    path = Path(str(value))

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def calculate_metrics(original, imputed, mask, numeric_columns):
    """Calculate RMSE and MAE at artificially hidden numeric cells."""
    original_values = original[numeric_columns].to_numpy(
        dtype=float
    )
    imputed_values = imputed[numeric_columns].to_numpy(
        dtype=float
    )
    hidden_mask = mask[numeric_columns].to_numpy(dtype=bool)

    # Evaluate only cells marked as artificially missing.
    valid = hidden_mask & np.isfinite(original_values)

    if not valid.any():
        return {
            "evaluated_hidden_numeric_cells": 0,
            "rmse": None,
            "mae": None,
        }

    errors = imputed_values[valid] - original_values[valid]

    return {
        "evaluated_hidden_numeric_cells": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
    }


# ============================================================
# 3. LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

print("=" * 65)
print("EMI PILOT EXPERIMENTS")
print("=" * 65)
print(f"Manifest: {MANIFEST_PATH}")
print(f"Experiments in manifest: {len(manifest)}")


# Find the manifest's column names.
dataset_col = find_column(
    manifest, ["dataset", "dataset_name"], "dataset"
)

pattern_col = find_column(
    manifest, ["pattern", "missingness_pattern"],
    "missingness pattern"
)

ratio_col = find_column(
    manifest, ["missing_ratio", "target_ratio", "ratio"],
    "missing ratio"
)

model_col = find_column(
    manifest, ["model", "missingness_model"],
    "missingness model"
)

rep_col = find_column(
    manifest, ["repetition", "rep", "repetition_id"],
    "repetition"
)

id_col = find_column(
    manifest, ["experiment_id", "id"], "experiment ID"
)

data_path_col = find_column(
    manifest,
    [
        "masked_data_file",
        "incomplete_path",
        "data_path",
        "output_path",
        "masked_data_path",
        "incomplete_file",
    ],
    "incomplete-data path",
)

mask_path_col = find_column(
    manifest,
    [
        "mask_file",
        "mask_path",
        "missingness_mask_path",
    ],
    "mask path",
)

# ============================================================
# 4. SELECT PILOT EXPERIMENTS
# ============================================================

# Pilot configuration:
# Pattern = simple
# Missing ratio = 1%
# Missingness model = overall
# Repetition = 1

ratios = pd.to_numeric(manifest[ratio_col], errors="coerce")
repetitions = pd.to_numeric(
    manifest[rep_col], errors="coerce"
)

ratio_matches = (
    np.isclose(ratios, 0.01)
    | np.isclose(ratios, 1.0)
)

selected = manifest[
    manifest[pattern_col]
    .astype(str).str.strip().str.lower().eq("simple")
    & ratio_matches
    & manifest[model_col]
    .astype(str).str.strip().str.lower().eq("overall")
    & repetitions.eq(1)
].copy()

if selected.empty:
    raise ValueError(
        "No experiments matched simple / 1% / overall / "
        "repetition 1.\n"
        "Check the actual values in your manifest."
    )

# Select only one experiment for each dataset.
selected = selected.drop_duplicates(subset=[dataset_col])

print(f"\nSelected pilot experiments: {len(selected)}")
print(
    selected[[dataset_col, id_col]].to_string(index=False)
)


# ============================================================
# 5. RUN EMI ON EACH PILOT EXPERIMENT
# ============================================================

summary = []

for _, experiment in selected.iterrows():

    dataset_name = str(experiment[dataset_col])
    experiment_id = str(experiment[id_col])

    print("\n" + "-" * 65)
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {experiment_id}")

    # Get the paths recorded in the manifest.
    incomplete_path = resolve_path(
        experiment[data_path_col]
    )
    mask_path = resolve_path(
        experiment[mask_path_col]
    )

    # Map each dataset name to its original cleaned file.
    normalized_name = dataset_name.strip().lower()

    if normalized_name == "adult":
        original_path = DATA_DIR / "adult_clean.csv"

    elif normalized_name in {
        "credit_approval",
        "credit",
        "creditapproval",
    }:
        original_path = DATA_DIR / "credit_clean.csv"

    else:
        raise ValueError(
            f"Unrecognized dataset name: {dataset_name}"
        )

    # Confirm all required input files exist.
    for path in [
        incomplete_path,
        mask_path,
        original_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load all data.
    masked = pd.read_csv(incomplete_path)
    original = pd.read_csv(original_path)
    mask = pd.read_csv(mask_path)

    # --------------------------------------------------------
    # Validate dimensions and columns
    # --------------------------------------------------------

    if masked.shape != original.shape:
        raise ValueError(
            f"Shape mismatch: incomplete={masked.shape}, "
            f"original={original.shape}"
        )

    if mask.shape != masked.shape:
        raise ValueError(
            f"Mask shape mismatch: data={masked.shape}, "
            f"mask={mask.shape}"
        )

    if list(masked.columns) != list(original.columns):
        raise ValueError(
            "Incomplete and original datasets have "
            "different column names or order."
        )

    if list(mask.columns) != list(masked.columns):
        raise ValueError(
            "Mask columns do not match the data columns."
        )

    # Convert saved mask values to Boolean values.
    mask = mask.apply(
        lambda column: column.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if mask.isna().any().any():
        raise ValueError(
            "The mask contains values that could not be "
            "converted to True or False."
        )

    mask = mask.astype(bool)

    # Verify the mask against the actual incomplete data.
    actual_missing = masked.isna().to_numpy()
    expected_missing = mask.to_numpy()

    if not np.array_equal(actual_missing, expected_missing):
        mismatch_count = int(
            np.sum(actual_missing != expected_missing)
        )

        raise ValueError(
            f"Mask validation failed for {experiment_id}. "
            f"Mismatched cells: {mismatch_count}"
        )

    # --------------------------------------------------------
    # Identify and prepare numeric columns
    # --------------------------------------------------------

    numeric_columns = original.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_columns:
        raise ValueError(
            f"No numeric columns found for {dataset_name}."
        )

    numeric_masked = masked[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Prevent unexpected text values from being silently
    # interpreted as missing data.
    conversion_errors = (
        masked[numeric_columns].notna()
        & numeric_masked.isna()
    )

    if conversion_errors.any().any():
        bad_columns = conversion_errors.columns[
            conversion_errors.any()
        ].tolist()

        raise ValueError(
            f"Unexpected non-numeric values in columns: "
            f"{bad_columns}"
        )

    print(f"Input shape: {masked.shape}")
    print(f"Numeric columns: {len(numeric_columns)}")
    print(
        f"Missing cells before imputation: "
        f"{int(actual_missing.sum())}"
    )

    # --------------------------------------------------------
    # Run the existing EMI-style imputer
    # --------------------------------------------------------

    numeric_imputed = emi_impute(numeric_masked)

    # Restore the imputed values into the full dataset.
    imputed = masked.copy()

    imputed.loc[:, numeric_columns] = numeric_imputed

    # --------------------------------------------------------
    # Validate imputation and calculate metrics
    # --------------------------------------------------------

    numeric_missing_after = int(
        imputed[numeric_columns].isna().sum().sum()
    )

    categorical_columns = [
        column
        for column in masked.columns
        if column not in numeric_columns
    ]

    if categorical_columns:
        categorical_missing_remaining = int(
            imputed[categorical_columns].isna().sum().sum()
        )
    else:
        categorical_missing_remaining = 0

    metrics = calculate_metrics(
        original=original,
        imputed=imputed,
        mask=mask,
        numeric_columns=numeric_columns,
    )

    # --------------------------------------------------------
    # Save the imputed dataset
    # --------------------------------------------------------

    output_path = RESULTS_DIR / (
        f"{experiment_id}_emi.csv.gz"
    )

    imputed.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    result = {
        "dataset": dataset_name,
        "experiment_id": experiment_id,
        "input_path": str(incomplete_path),
        "mask_path": str(mask_path),
        "output_path": str(output_path),
        "rows": int(imputed.shape[0]),
        "columns": int(imputed.shape[1]),
        "numeric_columns": len(numeric_columns),
        "missing_cells_before": int(actual_missing.sum()),
        "numeric_missing_after": numeric_missing_after,
        "categorical_missing_remaining": (
            categorical_missing_remaining
        ),
        "numeric_imputation_complete": (
            numeric_missing_after == 0
        ),
        **metrics,
    }

    summary.append(result)

    print(f"Saved output: {output_path}")
    print(
        f"Numeric missing values after EMI: "
        f"{numeric_missing_after}"
    )
    print(
        f"Categorical missing values remaining: "
        f"{categorical_missing_remaining}"
    )
    print(
        f"Hidden numeric cells evaluated: "
        f"{metrics['evaluated_hidden_numeric_cells']}"
    )
    print(f"RMSE: {metrics['rmse']}")
    print(f"MAE: {metrics['mae']}")
    print(
        f"Numeric imputation complete: "
        f"{numeric_missing_after == 0}"
    )


# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary_path = RESULTS_DIR / "pilot_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, allow_nan=False)

print("\n" + "=" * 65)
print("EMI PILOT FINISHED")
print("=" * 65)
print(f"Experiments completed: {len(summary)}")
print(f"Summary file: {summary_path}")
print(f"Results directory: {RESULTS_DIR}")

if all(
    item["numeric_imputation_complete"]
    for item in summary
):
    print(
        "PASS: All numeric missing values in the pilot "
        "experiments were imputed."
    )
else:
    print(
        "WARNING: Some numeric missing values remain. "
        "Review the output above."
    )

print(
    "\nResearch note: this runner uses the current "
    "emi_impute() function. The results should not yet "
    "be described as an exact reproduction of the paper's "
    "EMI algorithm."
)
