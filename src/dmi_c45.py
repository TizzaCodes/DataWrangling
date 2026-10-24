from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.emi import gaussian_emi


def _is_numeric(series: pd.Series) -> bool:
    """Return True if a pandas Series has a numeric dtype."""
    return pd.api.types.is_numeric_dtype(series)


def _make_target_categories(target: pd.Series) -> pd.Series:
    """
    Convert a numerical target into categories using quantile bins.

    The approximate number of categories is the square root of the
    number of unique observed target values.
    """
    observed = target.dropna()
    unique_count = observed.nunique()

    if unique_count <= 1:
        return pd.Series(0, index=observed.index, dtype=int)

    n_categories = min(
        unique_count,
        max(2, int(np.sqrt(unique_count))),
    )

    categories = pd.qcut(
        observed.rank(method="average"),
        q=n_categories,
        labels=False,
        duplicates="drop",
    )

    return categories.astype(int)


def _prepare_predictors(
    X: pd.DataFrame,
    numeric_fill_values: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare predictor columns for ChefBoost.

    Numeric predictors are discretized into quantile bins. All predictors
    are converted to ordinary Python strings stored with pandas object dtype.

    When fitting, bin edges and fill values are learned from the training
    records. The same edges and fill values are reused when routing records.
    """
    X = X.copy()

    numeric = [c for c in X.columns if _is_numeric(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    fitting = numeric_fill_values is None

    if fitting:
        numeric_fill_values = {}

    for column in numeric:
        values = pd.to_numeric(X[column], errors="coerce")

        if fitting:
            median = values.median()
            fill_value = float(median) if pd.notna(median) else 0.0
            numeric_fill_values[column] = fill_value
        else:
            fill_value = float(numeric_fill_values.get(column, 0.0))

        values = values.fillna(fill_value).astype(float)

        edges_key = f"__edges__{column}"

        if fitting:
            unique_count = int(values.nunique())

            if unique_count <= 1:
                edges = [-np.inf, np.inf]
            else:
                n_bins = min(
                    10,
                    max(2, int(np.sqrt(unique_count))),
                )

                try:
                    quantiles = np.linspace(0, 1, n_bins + 1)
                    inner_edges = np.unique(
                        values.quantile(quantiles).to_numpy()
                    )

                    edges = np.concatenate(
                        (
                            [-np.inf],
                            inner_edges[1:-1],
                            [np.inf],
                        )
                    ).tolist()

                    edges = sorted(set(float(edge) for edge in edges))

                    if len(edges) < 2:
                        edges = [-np.inf, np.inf]
                    else:
                        edges[0] = -np.inf
                        edges[-1] = np.inf

                except (ValueError, TypeError, IndexError):
                    edges = [-np.inf, np.inf]

            numeric_fill_values[edges_key] = edges

        else:
            edges = numeric_fill_values.get(
                edges_key,
                [-np.inf, np.inf],
            )

        binned = pd.cut(
            values,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )

        X[column] = (
            binned.astype("string")
            .fillna("__DMI_NUM_MISSING__")
            .astype(str)
        )

    for column in categorical:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("__DMI_MISSING__")
            .astype(str)
        )

    # Explicitly use object dtype. Some ChefBoost versions do not reliably
    # recognize pandas StringDtype columns as categorical.
    for column in X.columns:
        X[column] = pd.Series(
            [str(value) for value in X[column].tolist()],
            index=X.index,
            dtype=object,
        )

    # ChefBoost generates Python source code from feature names.
    # Safe feature names avoid issues with spaces or punctuation.
    X.columns = [
        f"feature_{i}"
        for i in range(X.shape[1])
    ]

    return X, numeric_fill_values


class _ReturnToLeaf(ast.NodeTransformer):
    """
    Change terminal return statements in ChefBoost's generated rules
    into integer leaf IDs.
    """

    def __init__(self):
        self.leaf_id = 0

    def visit_Return(self, node):
        leaf_id = self.leaf_id
        self.leaf_id += 1

        return ast.copy_location(
            ast.Return(value=ast.Constant(value=leaf_id)),
            node,
        )


class _C45Router:
    """Route records through a trained ChefBoost tree and return leaf IDs."""

    def __init__(
        self,
        find_decision,
        predictors: list[str],
        numeric_fill_values: dict[str, Any],
    ):
        self.find_decision = find_decision
        self.predictors = predictors
        self.numeric_fill_values = numeric_fill_values
        self.n_args = find_decision.__code__.co_argcount

    def apply(self, records: pd.DataFrame) -> pd.Series:
        if records.empty:
            return pd.Series(
                index=records.index,
                dtype="int64",
                name="leaf_id",
            )

        prepared, _ = _prepare_predictors(
            records[self.predictors],
            self.numeric_fill_values,
        )

        leaves = []

        for values in prepared.itertuples(
            index=False,
            name=None,
        ):
            values = list(values)

            if self.n_args == 1 and len(values) != 1:
                leaf_id = self.find_decision(values)
            else:
                leaf_id = self.find_decision(*values)

            leaves.append(int(leaf_id))

        return pd.Series(
            leaves,
            index=records.index,
            name="leaf_id",
        )


def _train_c45_tree(
    complete_records: pd.DataFrame,
    target_column: str,
) -> tuple[_C45Router | None, list[str], Any | None, dict[str, Any]]:
    """
    Train a ChefBoost C4.5 classifier.

    Numerical target values are discretized into classes for tree training.
    The generated decision rules are then adapted to expose terminal-leaf IDs.
    """
    predictors = [
        column
        for column in complete_records.columns
        if column != target_column
    ]

    if complete_records.empty or not predictors:
        return None, predictors, None, {}

    target = complete_records[target_column]

    if _is_numeric(target):
        classes = _make_target_categories(target).astype(str)
    else:
        classes = target.astype(str)

    if classes.nunique() < 2:
        modes = target.mode()
        fallback = modes.iloc[0] if not modes.empty else None
        return None, predictors, fallback, {}

    X, fill_values = _prepare_predictors(
        complete_records[predictors]
    )

    chef_data = X.copy()

    safe_target = "__dmi_target__"

    # FIX 1: explicitly enforce object dtype for the target.
    # This is important with newer pandas versions and ChefBoost 0.0.19.
    target_labels = classes.reindex(complete_records.index)

    chef_data = chef_data.astype("object")

    chef_data[safe_target] = pd.Series(
        target_labels.astype("object").tolist(),
        index=chef_data.index,
        dtype="object",
    )

    # Confirm that the target is really stored as object dtype.
    if chef_data[safe_target].dtype != object:
        raise TypeError(
            "ChefBoost target column must have object dtype; "
            f"got {chef_data[safe_target].dtype!r}."
        )

    try:
        from chefboost import Chefboost as chef
    except ImportError as exc:
        raise ImportError(
            "ChefBoost is required. Install it using: "
            "python -m pip install chefboost"
        ) from exc

    old_cwd = os.getcwd()
    old_sys_path = sys.path.copy()

    find_decision = None

    try:
        # FIX 2: ignore cleanup errors caused by temporary files being
        # locked on Windows. This does not affect tree training itself.
        with tempfile.TemporaryDirectory(
            prefix="dmi_c45_",
            ignore_cleanup_errors=True,
        ) as temp_dir:

            os.chdir(temp_dir)

            if temp_dir not in sys.path:
                sys.path.insert(0, temp_dir)

            chef.fit(
                chef_data,
                config={
                    "algorithm": "C4.5",
                    "enableParallelism": False,
                },
                target_label=safe_target,
            )

            rule_files = list(
                Path(temp_dir).rglob("rules.py")
            )

            rule_file = next(
                (
                    path
                    for path in rule_files
                    if "findDecision"
                    in path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                ),
                None,
            )

            if rule_file is None:
                raise RuntimeError(
                    "ChefBoost did not generate a rules.py file "
                    "containing findDecision."
                )

            source = rule_file.read_text(encoding="utf-8")
            tree_ast = ast.parse(source)

            tree_ast = _ReturnToLeaf().visit(tree_ast)
            ast.fix_missing_locations(tree_ast)

            namespace: dict[str, Any] = {}

            exec(
                compile(
                    tree_ast,
                    str(rule_file),
                    "exec",
                ),
                namespace,
            )

            find_decision = namespace.get("findDecision")

            if find_decision is None:
                raise RuntimeError(
                    "The generated C4.5 rules do not contain "
                    "a findDecision function."
                )

    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    router = _C45Router(
        find_decision,
        predictors,
        fill_values,
    )

    return router, predictors, classes, fill_values


def dmi_impute(
    data: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-5,
    eigen_floor: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values using a DMI-style procedure with ChefBoost C4.5.

    Procedure:
    1. Identify records that are complete.
    2. Train one C4.5 classification tree per attribute with missing values.
    3. Route incomplete records to terminal leaves.
    4. Use majority-class imputation for categorical targets.
    5. Use Gaussian EMI within the relevant leaf-specific data for numeric
       targets, with a median fallback if EMI fails.

    Note: This is an approximation of the paper's method, not an exact
    reproduction of its original C4.5 and EMI implementations.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must contain at least one row.")

    original = data.copy(deep=True)
    result = data.copy(deep=True)

    missing = original.isna()

    complete = original.loc[
        ~missing.any(axis=1)
    ].copy()

    numeric_columns = [
        column
        for column in original.columns
        if _is_numeric(original[column])
    ]

    diagnostics = {
        "rows": len(original),
        "columns": len(original.columns),
        "complete_training_records": len(complete),
        "missing_cells_before": int(missing.sum().sum()),
        "missing_cells_after": None,
        "attributes_processed": [],
        "emi_calls": 0,
        "tree_algorithm": "ChefBoost C4.5",
        "tree_fallbacks": [],
        "emi_diagnostics": [],
    }

    if not missing.to_numpy().any():
        diagnostics["missing_cells_after"] = 0
        return result, diagnostics

    for target in original.columns:
        missing_idx = original.index[missing[target]]

        if len(missing_idx) == 0:
            continue

        print(
            f"[DMI-C4.5] Target: {target!r} | "
            f"missing: {len(missing_idx)} | "
            f"complete training rows: {len(complete):,}",
            flush=True,
        )

        diagnostics["attributes_processed"].append(target)

        is_numeric_target = target in numeric_columns
        observed = original[target].dropna()

        if is_numeric_target:
            fallback = (
                float(observed.median())
                if not observed.empty
                else np.nan
            )
        else:
            modes = observed.mode()
            fallback = (
                modes.iloc[0]
                if not modes.empty
                else np.nan
            )

        tree, predictors, _, _ = _train_c45_tree(
            complete,
            target,
        )

        if tree is None:
            result.loc[missing_idx, target] = fallback

            diagnostics["tree_fallbacks"].append({
                "attribute": target,
                "reason": (
                    "No usable C4.5 tree; "
                    "used observed-value fallback."
                ),
            })

            continue

        training_leaves = tree.apply(complete)

        missing_records = original.loc[missing_idx].copy()
        missing_leaves = tree.apply(missing_records)

        for leaf_id in missing_leaves.unique():
            target_idx = missing_leaves.index[
                missing_leaves == leaf_id
            ]

            train_idx = training_leaves.index[
                training_leaves == leaf_id
            ]

            if len(train_idx) == 0:
                result.loc[target_idx, target] = fallback

                diagnostics["tree_fallbacks"].append({
                    "attribute": target,
                    "leaf_id": int(leaf_id),
                    "reason": (
                        "No complete training records in "
                        "the routed leaf."
                    ),
                })

                continue

            # Categorical target: assign the majority class from
            # complete training records in the same leaf.
            if not is_numeric_target:
                values = complete.loc[
                    train_idx,
                    target,
                ].dropna()

                modes = values.mode()

                result.loc[target_idx, target] = (
                    modes.iloc[0]
                    if not modes.empty
                    else fallback
                )

                continue

            # Numeric target: apply Gaussian EMI to records in the leaf.
            # Include complete training records and the incomplete records
            # being imputed in this leaf.
            leaf_data = pd.concat(
                [
                    original.loc[train_idx],
                    original.loc[target_idx],
                ],
                axis=0,
            )

            try:
                imputed, emi_info = gaussian_emi(
                    leaf_data,
                    max_iter=max_iter,
                    tol=tol,
                    eigen_floor=eigen_floor,
                )

                diagnostics["emi_calls"] += 1

                diagnostics["emi_diagnostics"].append({
                    "attribute": target,
                    "leaf_id": int(leaf_id),
                    "records": len(leaf_data),
                    "diagnostics": emi_info,
                })

                result.loc[target_idx, target] = imputed.loc[
                    target_idx,
                    target,
                ]

            except (
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                result.loc[target_idx, target] = fallback

                diagnostics["tree_fallbacks"].append({
                    "attribute": target,
                    "leaf_id": int(leaf_id),
                    "reason": f"Gaussian EMI failed: {exc}",
                })

    diagnostics["missing_cells_after"] = int(
        result.isna().sum().sum()
    )

    return result, diagnostics


def evaluate_numeric_imputation(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict:
    """
    Evaluate numerical imputation only at artificially hidden cells.

    The mask should be True for cells deliberately hidden during the
    missingness-generation step. Only cells with known original and
    imputed values are evaluated.
    """
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
    )

    per_column = {}
    all_actual = []
    all_predicted = []

    for column in original.columns:
        if not _is_numeric(original[column]):
            continue

        hidden = (
            mask[column].astype(bool)
            & original[column].notna()
            & imputed[column].notna()
        )

        if not hidden.any():
            continue

        actual = (
            original.loc[hidden, column]
            .astype(float)
            .to_numpy()
        )

        predicted = (
            imputed.loc[hidden, column]
            .astype(float)
            .to_numpy()
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        )

        mae = float(
            np.mean(np.abs(actual - predicted))
        )

        if len(actual) >= 2 and np.var(actual) > 0:
            r2 = float(r2_score(actual, predicted))
        else:
            r2 = float("nan")

        actual_mean = np.mean(actual)

        denominator = np.sum(
            (
                np.abs(predicted - actual_mean)
                + np.abs(actual - actual_mean)
            ) ** 2
        )

        numerator = np.sum(
            (predicted - actual) ** 2
        )

        d2 = (
            float(1 - numerator / denominator)
            if denominator > 0
            else float("nan")
        )

        per_column[column] = {
            "n": int(len(actual)),
            "RMSE": rmse,
            "R2": r2,
            "d2": d2,
            "MAE": mae,
        }

        all_actual.extend(actual.tolist())
        all_predicted.extend(predicted.tolist())

    if not all_actual:
        return {
            "per_column": {},
            "pooled": {},
            "n_evaluated": 0,
        }

    actual_all = np.asarray(
        all_actual,
        dtype=float,
    )

    predicted_all = np.asarray(
        all_predicted,
        dtype=float,
    )

    pooled_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_all,
                predicted_all,
            )
        )
    )

    pooled_mae = float(
        np.mean(
            np.abs(actual_all - predicted_all)
        )
    )

    if len(actual_all) >= 2 and np.var(actual_all) > 0:
        pooled_r2 = float(
            r2_score(actual_all, predicted_all)
        )
    else:
        pooled_r2 = float("nan")

    actual_mean = np.mean(actual_all)

    pooled_denominator = np.sum(
        (
            np.abs(predicted_all - actual_mean)
            + np.abs(actual_all - actual_mean)
        ) ** 2
    )

    pooled_numerator = np.sum(
        (predicted_all - actual_all) ** 2
    )

    pooled_d2 = (
        float(
            1 - pooled_numerator / pooled_denominator
        )
        if pooled_denominator > 0
        else float("nan")
    )

    return {
        "per_column": per_column,
        "pooled": {
            "RMSE": pooled_rmse,
            "R2": pooled_r2,
            "d2": pooled_d2,
            "MAE": pooled_mae,
        },
        "n_evaluated": len(actual_all),
    }