
"""
Resumable DMI vs Gaussian EMI evaluation suite.

Default output:
    results/dmi/full_suite/

Examples:
    One full-dataset experiment:
        python run_dmi_suite.py --limit 1 --rows 0

    All 640 full-dataset experiments:
        python run_dmi_suite.py --limit 640 --rows 0 --resume

    Resume an interrupted full run:
        python run_dmi_suite.py --limit 640 --rows 0 --resume

    Save each imputed dataset as well:
        python run_dmi_suite.py --limit 640 --rows 0 --resume --save-imputed
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dmi import dmi_impute
from src.emi import gaussian_emi


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "missingness" / "manifest.csv"

CLEAN_PATHS = {
    "adult": ROOT / "data" / "processed" / "adult_clean.csv",
    "credit_approval": ROOT / "data" / "processed" / "credit_clean.csv",
}

DEFAULT_OUTPUT = ROOT / "results" / "dmi" / "full_suite"


# ------------------------------------------------------------
# File and manifest helpers
# ------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, compression="infer")


def resolve_path(value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def find_column(df: pd.DataFrame, aliases: list[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(
        f"Missing manifest column; expected one of {aliases}. "
        f"Available columns: {list(df.columns)}"
    )


def map_manifest_columns(df: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "experiment_id": ["experiment_id"],
        "dataset": ["dataset"],
        "data_file": ["masked_data_file", "data_file"],
        "mask_file": ["mask_file"],
        "pattern": ["pattern", "missingness_pattern"],
        "target_ratio": ["target_ratio"],
        "actual_missing_ratio": ["actual_missing_ratio"],
        "model": ["model", "distribution_model"],
        "repetition": ["repetition", "rep"],
    }
    result = {}
    for key, names in aliases.items():
        try:
            result[key] = find_column(df, names)
        except ValueError:
            if key in {"experiment_id", "dataset", "data_file", "mask_file"}:
                raise
    return result


def load_mask(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    mask = read_csv(path)

    if list(mask.columns) != expected_columns:
        raise ValueError(
            f"Mask/data columns differ for {path}.\n"
            f"Mask: {list(mask.columns)}\nData: {expected_columns}"
        )

    out = pd.DataFrame(index=mask.index)

    for col in mask.columns:
        values = mask[col]

        if pd.api.types.is_bool_dtype(values):
            out[col] = values.fillna(False)

        elif pd.api.types.is_numeric_dtype(values):
            unique = set(values.dropna().unique())
            if not unique.issubset({0, 1}):
                raise ValueError(f"Mask {col} has non-binary values: {unique}")
            out[col] = values.fillna(0).astype(int).astype(bool)

        else:
            normalized = values.astype(str).str.strip().str.lower()
            true_values = {"true", "1", "yes"}
            false_values = {"false", "0", "no"}
            unknown = set(normalized.unique()) - true_values - false_values
            if unknown:
                raise ValueError(f"Unknown mask values in {col}: {unknown}")
            out[col] = normalized.isin(true_values)

    return out.astype(bool)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

def numeric_metrics(
    original: pd.DataFrame,
    result: pd.DataFrame,
    mask: pd.DataFrame,
    method: str,
    experiment_id: str,
) -> list[dict]:
    records = []

    for col in original.columns:
        if not is_numeric(original[col]):
            continue

        hidden = mask[col].to_numpy(dtype=bool)
        actual_all = pd.to_numeric(
            original[col], errors="coerce"
        ).to_numpy(dtype=float)
        predicted_all = pd.to_numeric(
            result[col], errors="coerce"
        ).to_numpy(dtype=float)

        valid = hidden & np.isfinite(actual_all) & np.isfinite(predicted_all)
        if not valid.any():
            continue

        actual = actual_all[valid]
        predicted = predicted_all[valid]
        errors = predicted - actual
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        sst = float(np.sum((actual - actual.mean()) ** 2))
        r2 = float(1 - np.sum(errors ** 2) / sst) if sst > 0 else np.nan

        if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
            corr = float(np.corrcoef(actual, predicted)[0, 1])
            squared_corr = corr ** 2
        else:
            squared_corr = np.nan

        d2_den = float(np.sum((
            np.abs(predicted - actual.mean())
            + np.abs(actual - actual.mean())
        ) ** 2))
        d2 = (
            float(1 - np.sum(errors ** 2) / d2_den)
            if d2_den > 0 else np.nan
        )

        clean_values = pd.to_numeric(
            original[col], errors="coerce"
        ).to_numpy(dtype=float)
        clean_values = clean_values[np.isfinite(clean_values)]
        value_range = (
            float(clean_values.max() - clean_values.min())
            if len(clean_values) else np.nan
        )
        nrmse = rmse / value_range if value_range > 0 else np.nan

        records.append({
            "experiment_id": experiment_id,
            "method": method,
            "attribute": col,
            "n_evaluated": int(valid.sum()),
            "RMSE": rmse,
            "NRMSE_minmax": nrmse,
            "MAE": mae,
            "R2_standard": r2,
            "squared_Pearson_correlation": squared_corr,
            "d2": d2,
        })

    return records


def pooled_metrics(
    original: pd.DataFrame,
    result: pd.DataFrame,
    mask: pd.DataFrame,
    method: str,
    experiment_id: str,
) -> dict:
    actual_parts = []
    predicted_parts = []
    normalized_errors = []
    n_attributes = 0

    for col in original.columns:
        if not is_numeric(original[col]):
            continue

        hidden = mask[col].to_numpy(dtype=bool)
        actual_all = pd.to_numeric(
            original[col], errors="coerce"
        ).to_numpy(dtype=float)
        predicted_all = pd.to_numeric(
            result[col], errors="coerce"
        ).to_numpy(dtype=float)

        valid = hidden & np.isfinite(actual_all) & np.isfinite(predicted_all)
        if not valid.any():
            continue

        actual = actual_all[valid]
        predicted = predicted_all[valid]
        actual_parts.append(actual)
        predicted_parts.append(predicted)
        n_attributes += 1

        clean = pd.to_numeric(
            original[col], errors="coerce"
        ).to_numpy(dtype=float)
        clean = clean[np.isfinite(clean)]
        if len(clean):
            value_range = float(clean.max() - clean.min())
            if value_range > 0:
                normalized_errors.extend(
                    ((predicted - actual) / value_range).tolist()
                )

    if not actual_parts:
        return {
            "experiment_id": experiment_id,
            "method": method,
            "n_evaluated": 0,
            "numeric_attributes_evaluated": 0,
            "RMSE_pooled_raw": np.nan,
            "MAE_pooled_raw": np.nan,
            "NRMSE_minmax_pooled": np.nan,
            "R2_standard_pooled": np.nan,
            "d2_pooled": np.nan,
        }

    actual = np.concatenate(actual_parts)
    predicted = np.concatenate(predicted_parts)
    errors = predicted - actual
    sse = float(np.sum(errors ** 2))
    sst = float(np.sum((actual - actual.mean()) ** 2))

    d2_den = float(np.sum((
        np.abs(predicted - actual.mean())
        + np.abs(actual - actual.mean())
    ) ** 2))

    return {
        "experiment_id": experiment_id,
        "method": method,
        "n_evaluated": int(len(actual)),
        "numeric_attributes_evaluated": n_attributes,
        "RMSE_pooled_raw": float(np.sqrt(np.mean(errors ** 2))),
        "MAE_pooled_raw": float(np.mean(np.abs(errors))),
        "NRMSE_minmax_pooled": (
            float(np.sqrt(np.mean(np.square(normalized_errors))))
            if normalized_errors else np.nan
        ),
        "R2_standard_pooled": (
            float(1 - sse / sst) if sst > 0 else np.nan
        ),
        "d2_pooled": (
            float(1 - sse / d2_den) if d2_den > 0 else np.nan
        ),
    }


# ------------------------------------------------------------
# One experiment
# ------------------------------------------------------------

def run_one(
    exp: pd.Series,
    cmap: dict[str, str],
    row_limit: int,
    output_dir: Path,
    save_imputed: bool,
) -> tuple[list[dict], list[dict], dict]:

    exp_id = str(exp[cmap["experiment_id"]])
    dataset = str(exp[cmap["dataset"]]).strip().lower()

    if dataset not in CLEAN_PATHS:
        raise ValueError(f"Unexpected dataset in manifest: {dataset}")

    original = read_csv(CLEAN_PATHS[dataset])
    masked_path = resolve_path(exp[cmap["data_file"]])
    mask_path = resolve_path(exp[cmap["mask_file"]])
    masked = read_csv(masked_path)

    if list(original.columns) != list(masked.columns):
        raise ValueError(f"{exp_id}: original/masked column mismatch.")

    mask = load_mask(mask_path, list(masked.columns))

    if not (len(original) == len(masked) == len(mask)):
        raise ValueError(
            f"{exp_id}: row counts differ: original={len(original)}, "
            f"masked={len(masked)}, mask={len(mask)}"
        )

    if row_limit > 0:
        original = original.iloc[:row_limit].reset_index(drop=True)
        masked = masked.iloc[:row_limit].reset_index(drop=True)
        mask = mask.iloc[:row_limit].reset_index(drop=True)

    hidden_count = int(mask.to_numpy(dtype=bool).sum())
    missing_count = int(masked.isna().to_numpy().sum())
    hidden_missing = int(
        (mask.to_numpy(dtype=bool) & masked.isna().to_numpy()).sum()
    )

    if hidden_missing != hidden_count:
        raise ValueError(
            f"{exp_id}: mask says {hidden_count} cells were hidden, "
            f"but only {hidden_missing} of those are missing in the input. "
            "Stopping to avoid invalid evaluation."
        )

    print(f"\n[{exp_id}] {dataset}: {len(masked):,} rows x {len(masked.columns)} cols")
    print(f"Masked cells: {hidden_count:,}")

    # DMI
    start = time.time()
    dmi_result, dmi_diag = dmi_impute(masked.copy())
    dmi_seconds = time.time() - start

    if dmi_result.shape != original.shape:
        raise ValueError(f"{exp_id}: DMI output shape changed.")
    if list(dmi_result.columns) != list(original.columns):
        raise ValueError(f"{exp_id}: DMI output columns changed.")

    dmi_remaining = int(dmi_result.isna().to_numpy().sum())

    # Ensure values not marked as hidden were preserved.
    observed = ~mask.to_numpy(dtype=bool)
    before = original.to_numpy(dtype=object)
    after = dmi_result.to_numpy(dtype=object)
    observed_changed = 0

    for i, j in zip(*np.where(observed)):
        a, b = before[i, j], after[i, j]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or a != b:
            observed_changed += 1

    # EMI numeric baseline
    start = time.time()
    emi_result, emi_diag = gaussian_emi(masked.copy())
    emi_seconds = time.time() - start
    emi_remaining = int(emi_result.isna().to_numpy().sum())

    metric_rows = []
    pooled_rows = []

    for method, result in [("DMI", dmi_result), ("EMI", emi_result)]:
        metric_rows.extend(
            numeric_metrics(original, result, mask, method, exp_id)
        )
        pooled_rows.append(
            pooled_metrics(original, result, mask, method, exp_id)
        )

    # Attach experiment metadata to metric records.
    metadata = {
        "dataset": dataset,
        "rows_used": len(masked),
    }
    for field in [
        "pattern", "target_ratio", "actual_missing_ratio", "model", "repetition"
    ]:
        if field in cmap:
            metadata[field] = exp[cmap[field]]

    for row in metric_rows + pooled_rows:
        row.update(metadata)

    # Optional: preserve imputed datasets if explicitly requested.
    dmi_path = ""
    emi_path = ""
    if save_imputed:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in exp_id
        )
        dmi_path = str((output_dir / f"{safe_id}_dmi.csv.gz").relative_to(ROOT))
        emi_path = str((output_dir / f"{safe_id}_emi.csv.gz").relative_to(ROOT))
        dmi_result.to_csv(ROOT / dmi_path, index=False, compression="gzip")
        emi_result.to_csv(ROOT / emi_path, index=False, compression="gzip")

    diag_row = {
        "experiment_id": exp_id,
        **metadata,
        "mask_hidden_cells": hidden_count,
        "input_missing_cells": missing_count,
        "dmi_remaining_missing_cells": dmi_remaining,
        "dmi_observed_cells_changed": observed_changed,
        "dmi_seconds": dmi_seconds,
        "emi_remaining_missing_cells": emi_remaining,
        "emi_seconds": emi_seconds,
        "dmi_diagnostics_json": json.dumps(dmi_diag, default=str),
        "emi_diagnostics_json": json.dumps(emi_diag, default=str),
        "dmi_output_file": dmi_path,
        "emi_output_file": emi_path,
    }

    print(
        f"  DMI: {dmi_seconds:.1f}s, remaining missing={dmi_remaining}, "
        f"observed changed={observed_changed}"
    )
    print(f"  EMI: {emi_seconds:.1f}s, remaining missing={emi_remaining}")

    if dmi_remaining != 0:
        print(f"  WARNING: DMI left missing cells in {exp_id}.")

    if observed_changed != 0:
        raise ValueError(
            f"{exp_id}: DMI changed {observed_changed} observed cells. "
            "Stopping to protect the evaluation."
        )

    return metric_rows, pooled_rows, diag_row


# ------------------------------------------------------------
# Save checkpoints and summaries
# ------------------------------------------------------------

def write_outputs(
    output_dir: Path,
    attribute_rows: list[dict],
    pooled_rows: list[dict],
    diagnostic_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    attributes = pd.DataFrame(attribute_rows)
    pooled = pd.DataFrame(pooled_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    attributes.to_csv(output_dir / "per_attribute_metrics.csv", index=False)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    diagnostics.to_csv(output_dir / "experiment_diagnostics.csv", index=False)

    if not pooled.empty:
        group_cols = [
            col for col in [
                "dataset", "pattern", "target_ratio", "model", "method"
            ]
            if col in pooled.columns
        ]

        metric_cols = [
            col for col in [
                "RMSE_pooled_raw",
                "MAE_pooled_raw",
                "NRMSE_minmax_pooled",
                "R2_standard_pooled",
                "d2_pooled",
            ]
            if col in pooled.columns
        ]

        if group_cols and metric_cols:
            grouped = pooled.groupby(
                group_cols, dropna=False
            )[metric_cols]

            summary = grouped.agg(["mean", "std", "count"]).reset_index()
            summary.columns = [
                "_".join(str(part) for part in col if str(part))
                if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]
            summary.to_csv(output_dir / "summary_by_condition.csv", index=False)

    if not attributes.empty:
        group_cols = [
            col for col in [
                "dataset", "pattern", "target_ratio", "model",
                "method", "attribute"
            ]
            if col in attributes.columns
        ]
        metric_cols = [
            col for col in [
                "RMSE", "NRMSE_minmax", "MAE",
                "R2_standard", "squared_Pearson_correlation", "d2"
            ]
            if col in attributes.columns
        ]
        if group_cols and metric_cols:
            summary = attributes.groupby(
                group_cols, dropna=False
            )[metric_cols].agg(["mean", "std", "count"]).reset_index()
            summary.columns = [
                "_".join(str(part) for part in col if str(part))
                if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]
            summary.to_csv(
                output_dir / "summary_by_attribute.csv", index=False
            )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the DMI/EMI missing-value imputation suite."
    )
    parser.add_argument(
        "--limit", type=int, default=640,
        help="Maximum number of experiments to select from the manifest."
    )
    parser.add_argument(
        "--rows", type=int, default=0,
        help="Rows per experiment; 0 means all rows (default: 0)."
    )
    parser.add_argument(
        "--dataset", choices=["adult", "credit_approval"],
        help="Optionally run only one dataset."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT),
        help="Output directory."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip experiments already listed in the output diagnostics file."
    )
    parser.add_argument(
        "--save-imputed", action="store_true",
        help="Also save a compressed imputed CSV for every method/experiment."
    )
    args = parser.parse_args()

    if args.limit < 1 or args.rows < 0:
        parser.error("--limit must be positive and --rows must be non-negative.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    cmap = map_manifest_columns(manifest)

    if args.dataset:
        manifest = manifest[
            manifest[cmap["dataset"]].astype(str).str.strip().str.lower()
            == args.dataset
        ].copy()

    manifest = manifest.head(args.limit).reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No experiments selected.")

    attribute_file = output_dir / "per_attribute_metrics.csv"
    pooled_file = output_dir / "pooled_metrics.csv"
    diagnostics_file = output_dir / "experiment_diagnostics.csv"

    attribute_rows = (
        pd.read_csv(attribute_file).to_dict("records")
        if args.resume and attribute_file.exists() else []
    )
    pooled_rows = (
        pd.read_csv(pooled_file).to_dict("records")
        if args.resume and pooled_file.exists() else []
    )
    diagnostic_rows = (
        pd.read_csv(diagnostics_file).to_dict("records")
        if args.resume and diagnostics_file.exists() else []
    )

    completed = (
        set(pd.DataFrame(diagnostic_rows)["experiment_id"].astype(str))
        if diagnostic_rows else set()
    )

    selected_ids = set(manifest[cmap["experiment_id"]].astype(str))
    if args.resume:
        # Retain only results belonging to the selected manifest/dataset.
        attribute_rows = [
            r for r in attribute_rows
            if str(r.get("experiment_id")) in selected_ids
        ]
        pooled_rows = [
            r for r in pooled_rows
            if str(r.get("experiment_id")) in selected_ids
        ]
        diagnostic_rows = [
            r for r in diagnostic_rows
            if str(r.get("experiment_id")) in selected_ids
        ]
        completed &= selected_ids

    print(f"Project root: {ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Selected experiments: {len(manifest)}")
    print(f"Already completed: {len(completed)}")
    print(f"Rows: {'all' if args.rows == 0 else args.rows}")
    print(f"Output directory: {output_dir}")
    print(f"Save imputed datasets: {args.save_imputed}")

    started = time.time()
    newly_completed = 0

    for _, exp in manifest.iterrows():
        exp_id = str(exp[cmap["experiment_id"]])

        if args.resume and exp_id in completed:
            print(f"\nSkipping completed experiment: {exp_id}")
            continue

        attr, pooled, diag = run_one(
            exp=exp,
            cmap=cmap,
            row_limit=args.rows,
            output_dir=output_dir,
            save_imputed=args.save_imputed,
        )

        attribute_rows.extend(attr)
        pooled_rows.extend(pooled)
        diagnostic_rows.append(diag)
        newly_completed += 1

        # Checkpoint immediately after each completed experiment.
        write_outputs(
            output_dir, attribute_rows, pooled_rows, diagnostic_rows
        )

        elapsed = time.time() - started
        print(
            f"Checkpoint saved: {len(diagnostic_rows)} experiments recorded; "
            f"{elapsed / 60:.1f} minutes elapsed."
        )

    print("\n" + "=" * 72)
    print("SUITE FINISHED")
    print("=" * 72)
    print(f"Newly completed this run: {newly_completed}")
    print(f"Total recorded: {len(diagnostic_rows)}")
    print(f"Output directory: {output_dir}")
    print("Files:")
    for filename in [
        "per_attribute_metrics.csv",
        "pooled_metrics.csv",
        "experiment_diagnostics.csv",
        "summary_by_condition.csv",
        "summary_by_attribute.csv",
    ]:
        print(f"  {output_dir / filename}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
