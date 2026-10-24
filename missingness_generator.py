"""
Generate masked datasets for the missing-value imputation experiments.

Important methodological note:
The row-level definitions below preserve the operational definitions in the
project's existing generator. They should be checked against the paper before
describing the output as an exact replication.

Patterns:
- simple: at most one missing cell per row
- medium: 2..floor(n_columns / 2) missing cells in each affected row
- complex: ceil(0.50 * n_columns)..floor(0.80 * n_columns) missing cells
- blended: affected rows split approximately 25% simple, 50% medium,
  and 25% complex

Models:
- overall: randomly choose columns within each affected row
- UD: prefer columns with the fewest missing cells so far

The generator reaches the target number of missing cells when feasible.
For simple at 10%, the target is mathematically infeasible for these datasets
because at most one cell per row may be missing. In that case it records the
achieved count and ratio explicitly rather than silently violating the pattern.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


SEED = 42
RATIOS = [0.01, 0.03, 0.05, 0.10]
PATTERNS = ["simple", "medium", "complex", "blended"]
MODELS = ["overall", "UD"]
REPETITIONS = 10


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "data" / "missingness"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

DATASETS = {
    "adult": DATA_DIR / "adult_clean.csv",
    "credit_approval": DATA_DIR / "credit_clean.csv",
}


def row_limits(pattern, n_columns):
    """Inclusive minimum and maximum missing cells for an affected row."""
    if pattern == "simple":
        return 1, 1
    if pattern == "medium":
        return 2, max(2, n_columns // 2)
    if pattern == "complex":
        return int(np.ceil(0.50 * n_columns)), int(np.floor(0.80 * n_columns))
    raise ValueError(f"Unknown pattern: {pattern}")


def _pattern_counts(n_affected):
    """Return blended category counts, summing exactly to n_affected."""
    n_simple = int(round(0.25 * n_affected))
    n_medium = int(round(0.50 * n_affected))
    n_complex = n_affected - n_simple - n_medium
    return n_simple, n_medium, n_complex


def _find_feasible_affected_count(target_cells, n_rows, n_columns, pattern):
    """Find an affected-row count whose allowed degree range can meet target."""
    if target_cells <= 0:
        return 0

    if pattern == "simple":
        return min(target_cells, n_rows)

    candidates = []
    for n_affected in range(1, n_rows + 1):
        if pattern in ("medium", "complex"):
            low, high = row_limits(pattern, n_columns)
            min_total = n_affected * low
            max_total = n_affected * high
        elif pattern == "blended":
            ns, nm, nc = _pattern_counts(n_affected)
            s_low, s_high = row_limits("simple", n_columns)
            m_low, m_high = row_limits("medium", n_columns)
            c_low, c_high = row_limits("complex", n_columns)
            min_total = ns * s_low + nm * m_low + nc * c_low
            max_total = ns * s_high + nm * m_high + nc * c_high
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        if min_total <= target_cells <= max_total:
            candidates.append(n_affected)

    if not candidates:
        return None

    # Choose a feasible row count near the midpoint of each row's allowed range.
    def expected_total(n_affected):
        if pattern == "blended":
            ns, nm, nc = _pattern_counts(n_affected)
            s_low, s_high = row_limits("simple", n_columns)
            m_low, m_high = row_limits("medium", n_columns)
            c_low, c_high = row_limits("complex", n_columns)
            return (
                ns * (s_low + s_high) / 2
                + nm * (m_low + m_high) / 2
                + nc * (c_low + c_high) / 2
            )
        low, high = row_limits(pattern, n_columns)
        return n_affected * (low + high) / 2

    return min(candidates, key=lambda n: (abs(expected_total(n) - target_cells), n))


def _allocate_degrees_exact(target_cells, row_types, n_columns, rng):
    """Assign valid row degrees summing exactly to target_cells."""
    degrees = np.zeros(len(row_types), dtype=int)
    limits = [row_limits(t, n_columns) for t in row_types]
    minimum = sum(low for low, _ in limits)
    maximum = sum(high for _, high in limits)

    if not minimum <= target_cells <= maximum:
        raise ValueError(
            f"Cannot allocate {target_cells} cells under row constraints "
            f"(feasible range {minimum}..{maximum})."
        )

    for i, (low, _) in enumerate(limits):
        degrees[i] = low

    remaining = target_cells - minimum
    # Randomized order makes increments reproducible but not systematically
    # concentrated in the first rows.
    while remaining > 0:
        eligible = [i for i, (_, high) in enumerate(limits) if degrees[i] < high]
        if not eligible:
            raise RuntimeError("Degree allocation ended before reaching target.")
        rng.shuffle(eligible)
        for i in eligible:
            if remaining == 0:
                break
            degrees[i] += 1
            remaining -= 1

    return degrees


def make_row_degrees(n_rows, n_columns, target_cells, pattern, rng):
    """Choose row degrees, reaching target exactly whenever pattern permits."""
    degrees = np.zeros(n_rows, dtype=int)
    target_cells = int(target_cells)

    if pattern == "simple":
        # At most one missing cell per row, so this can cap below the target.
        count = min(target_cells, n_rows)
        selected = rng.permutation(n_rows)[:count]
        degrees[selected] = 1
        return degrees

    n_affected = _find_feasible_affected_count(
        target_cells, n_rows, n_columns, pattern
    )

    if n_affected is None:
        # Explicitly cap to the maximum achievable count under the pattern.
        if pattern in ("medium", "complex"):
            _, high = row_limits(pattern, n_columns)
            target_cells = min(target_cells, n_rows * high)
        else:
            # Blended maximum under the specified mixture.
            ns, nm, nc = _pattern_counts(n_rows)
            _, s_high = row_limits("simple", n_columns)
            _, m_high = row_limits("medium", n_columns)
            _, c_high = row_limits("complex", n_columns)
            target_cells = min(
                target_cells, ns * s_high + nm * m_high + nc * c_high
            )
        n_affected = _find_feasible_affected_count(
            target_cells, n_rows, n_columns, pattern
        )

    if n_affected is None:
        raise ValueError(
            f"Could not allocate target={target_cells} for pattern={pattern}."
        )

    selected_rows = rng.permutation(n_rows)[:n_affected]

    if pattern in ("medium", "complex"):
        row_types = [pattern] * n_affected
    else:
        ns, nm, nc = _pattern_counts(n_affected)
        row_types = ["simple"] * ns + ["medium"] * nm + ["complex"] * nc
        rng.shuffle(row_types)

    allocated = _allocate_degrees_exact(
        target_cells, row_types, n_columns, rng
    )
    degrees[selected_rows] = allocated
    return degrees


def make_mask(df, ratio, pattern, model, rng):
    """Create a boolean mask where True means that the cell is hidden."""
    n_rows, n_columns = df.shape
    target_cells = round(n_rows * n_columns * ratio)

    degrees = make_row_degrees(
        n_rows, n_columns, target_cells, pattern, rng
    )

    mask = np.zeros((n_rows, n_columns), dtype=bool)
    column_counts = np.zeros(n_columns, dtype=int)

    for row, degree in enumerate(degrees):
        if degree == 0:
            continue

        if model == "overall":
            selected = rng.choice(n_columns, size=degree, replace=False)
        elif model == "UD":
            tie_break = rng.random(n_columns)
            selected = np.lexsort((tie_break, column_counts))[:degree]
        else:
            raise ValueError(f"Unknown model: {model}")

        mask[row, selected] = True
        column_counts[selected] += 1

    return mask


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Cleaned source dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if df.isna().any().any():
            raise ValueError(
                f"{dataset_name} contains natural missing values. "
                "Check preprocessing before generating artificial missingness."
            )

        print(
            f"\nDataset: {dataset_name} | Rows: {len(df)} | "
            f"Columns: {len(df.columns)}"
        )

        for pattern in PATTERNS:
            for ratio in RATIOS:
                for model in MODELS:
                    for repetition in range(1, REPETITIONS + 1):
                        # Each experiment gets a deterministic independent stream.
                        seed_sequence = np.random.SeedSequence(
                            [SEED, list(DATASETS).index(dataset_name),
                             PATTERNS.index(pattern), RATIOS.index(ratio),
                             MODELS.index(model), repetition]
                        )
                        rng = np.random.default_rng(seed_sequence)

                        mask = make_mask(df, ratio, pattern, model, rng)
                        mask_df = pd.DataFrame(mask, columns=df.columns)
                        incomplete = df.mask(mask_df)

                        ratio_label = f"{int(ratio * 100):02d}pct"
                        experiment_id = (
                            f"{dataset_name}_{pattern}_{ratio_label}_"
                            f"{model}_rep{repetition:02d}"
                        )

                        folder = OUTPUT_DIR / dataset_name
                        folder.mkdir(parents=True, exist_ok=True)
                        data_path = folder / f"{experiment_id}.csv.gz"
                        mask_path = folder / f"{experiment_id}_mask.csv.gz"

                        incomplete.to_csv(
                            data_path, index=False, compression="gzip"
                        )
                        mask_df.to_csv(
                            mask_path, index=False, compression="gzip"
                        )

                        actual_cells = int(mask.sum())
                        total_cells = int(mask.size)
                        expected_target = round(total_cells * ratio)

                        # Validate the saved artifacts, not just in-memory objects.
                        saved_data = pd.read_csv(data_path)
                        saved_mask = pd.read_csv(mask_path).astype(bool)
                        saved_missing = int(saved_data.isna().sum().sum())
                        saved_mask_count = int(saved_mask.to_numpy().sum())

                        if saved_data.shape != df.shape or saved_mask.shape != df.shape:
                            raise RuntimeError(
                                f"Shape validation failed for {experiment_id}"
                            )
                        if saved_missing != saved_mask_count:
                            raise RuntimeError(
                                f"Mask/data mismatch for {experiment_id}: "
                                f"{saved_missing} missing values vs "
                                f"{saved_mask_count} mask entries"
                            )

                        manifest.append({
                            "experiment_id": experiment_id,
                            "dataset": dataset_name,
                            "pattern": pattern,
                            "target_ratio": ratio,
                            "model": model,
                            "repetition": repetition,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "target_missing_cells": expected_target,
                            "actual_missing_cells": actual_cells,
                            "actual_missing_ratio": actual_cells / total_cells,
                            "target_reached": actual_cells == expected_target,
                            "masked_data_file": str(data_path.relative_to(PROJECT_DIR)),
                            "mask_file": str(mask_path.relative_to(PROJECT_DIR)),
                        })

        print(f"Finished generating {dataset_name}.")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    print("\nGENERATION COMPLETE")
    print(f"Experiments generated: {len(manifest_df)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("\nMean achieved missingness by pattern, ratio, and model:")
    print(
        manifest_df.groupby(["pattern", "target_ratio", "model"])
        ["actual_missing_ratio"].mean().round(4)
    )
    not_reached = manifest_df.loc[~manifest_df["target_reached"]]
    if not not_reached.empty:
        print("\nNOTE: Some targets were infeasible under the pattern constraints:")
        print(
            not_reached[
                ["experiment_id", "target_missing_cells",
                 "actual_missing_cells", "actual_missing_ratio"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    generate_all()
