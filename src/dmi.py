
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from src.emi import gaussian_emi


def _is_numeric(series: pd.Series) -> bool:
    """Check whether a column has a numerical data type."""
    return pd.api.types.is_numeric_dtype(series)


def _make_target_categories(target: pd.Series) -> pd.Series:
    """
    Categorise a numerical target for decision-tree training.

    The paper describes using approximately the square root of the
    target's domain size. Quantile categories are an implementation
    choice because the paper does not specify the binning procedure.
    """
    observed = target.dropna()
    unique_count = observed.nunique()

    if unique_count <= 1:
        return pd.Series(0, index=observed.index, dtype=int)

    n_categories = min(
        unique_count,
        max(2, int(np.sqrt(unique_count))),
    )

    ranks = observed.rank(method="average")

    categories = pd.qcut(
        ranks,
        q=n_categories,
        labels=False,
        duplicates="drop",
    )

    return categories.astype(int)


def _encode_predictors(
    X: pd.DataFrame,
    encoder: ColumnTransformer | None = None,
    numeric_fill_values: dict | None = None,
) -> tuple[np.ndarray, ColumnTransformer | None, dict]:
    """
    Encode predictors using a consistent fitted encoder.

    Numeric missing predictors are filled using medians calculated
    from the complete training records. Categorical predictors use
    an explicit missing-category marker.

    These fills are for tree routing only; they do not replace the
    final DMI imputation.
    """
    X_encoded = X.copy()

    numeric_columns = [
        column for column in X.columns
        if _is_numeric(X[column])
    ]
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    if numeric_fill_values is None:
        numeric_fill_values = {}

        for column in numeric_columns:
            values = pd.to_numeric(
                X_encoded[column], errors="coerce"
            )
            median = values.median()
            numeric_fill_values[column] = (
                float(median) if pd.notna(median) else 0.0
            )

    for column in numeric_columns:
        X_encoded[column] = (
            pd.to_numeric(X_encoded[column], errors="coerce")
            .fillna(numeric_fill_values.get(column, 0.0))
            .astype(float)
        )

    for column in categorical_columns:
        X_encoded[column] = (
            X_encoded[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    if encoder is None:
        transformers = []

        if numeric_columns:
            transformers.append(
                ("numeric", "passthrough", numeric_columns)
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    categorical_columns,
                )
            )

        if not transformers:
            return np.empty((len(X_encoded), 0)), None, numeric_fill_values

        encoder = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        encoded = encoder.fit_transform(X_encoded)
    else:
        encoded = encoder.transform(X_encoded)

    return encoded, encoder, numeric_fill_values


def _train_attribute_tree(
    complete_records: pd.DataFrame,
    target_column: str,
) -> tuple[
    DecisionTreeClassifier | None,
    list[str],
    object | None,
    ColumnTransformer | None,
    dict,
]:
    """
    Train one entropy-based decision tree for a target attribute.

    Only records complete across all attributes are used for training,
    following the paper's general procedure.

    Returns:
        tree, predictor_names, encoded_target, fitted_encoder,
        numeric_predictor_fill_values
    """
    predictors = [
        column for column in complete_records.columns
        if column != target_column
    ]

    if complete_records.empty or not predictors:
        return None, predictors, None, None, {}

    target = complete_records[target_column]

    if _is_numeric(target):
        encoded_target = _make_target_categories(target)
    else:
        encoded_target = target.astype(str)

    if encoded_target.nunique() < 2:
        fallback = target.mode().iloc[0] if not target.mode().empty else None
        return None, predictors, fallback, None, {}

    X = complete_records[predictors]

    X_encoded, encoder, numeric_fill_values = _encode_predictors(X)

    tree = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
    )

    tree.fit(X_encoded, encoded_target)

    return (
        tree,
        predictors,
        encoded_target,
        encoder,
        numeric_fill_values,
    )


def _assign_records_to_leaves(
    tree: DecisionTreeClassifier,
    predictors: list[str],
    records: pd.DataFrame,
    encoder: ColumnTransformer,
    numeric_fill_values: dict,
) -> pd.Series:
    """
    Assign records to terminal leaves using the fitted training encoder.

    Original row indices are preserved.
    """
    if records.empty:
        return pd.Series(index=records.index, dtype="int64", name="leaf_id")

    X = records[predictors].copy()

    X_encoded, _, _ = _encode_predictors(
        X,
        encoder=encoder,
        numeric_fill_values=numeric_fill_values,
    )

    leaf_ids = tree.apply(X_encoded)

    return pd.Series(
        leaf_ids,
        index=records.index,
        name="leaf_id",
    )


def dmi_impute(
    data: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-5,
    eigen_floor: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a DMI-style imputation procedure.

    For every attribute with missing values:
      1. Train a tree using complete records.
      2. Assign records missing that target to tree leaves.
      3. For numerical targets, run Gaussian EMI on each relevant
         leaf-specific dataset and use the imputed target values.
      4. For categorical targets, use the majority target class
         among complete training records in the assigned leaf.
      5. Use a global observed-value fallback if no usable tree exists.

    This is a documented approximation of the paper, not an exact
    C4.5 reproduction.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must contain at least one row.")

    original = data.copy(deep=True)
    result = data.copy(deep=True)

    original_missing = original.isna()
    complete_records = original.loc[~original_missing.any(axis=1)].copy()

    numeric_columns = [
        column for column in original.columns
        if _is_numeric(original[column])
    ]

    diagnostics = {
        "rows": len(original),
        "columns": len(original.columns),
        "complete_training_records": len(complete_records),
        "missing_cells_before": int(original_missing.sum().sum()),
        "missing_cells_after": None,
        "attributes_processed": [],
        "emi_calls": 0,
        "tree_fallbacks": [],
        "emi_diagnostics": [],
    }

    if original_missing.sum().sum() == 0:
        diagnostics["missing_cells_after"] = 0
        return result, diagnostics

    for target_column in original.columns:
        missing_indices = original.index[
            original_missing[target_column]
        ]

        if len(missing_indices) == 0:
            continue

        diagnostics["attributes_processed"].append(target_column)

        is_numeric_target = target_column in numeric_columns

        # Fallback values use observed target values, never the hidden
        # values from the original complete reference dataset.
        observed_target = original[target_column].dropna()

        if is_numeric_target:
            fallback = (
                float(observed_target.median())
                if not observed_target.empty
                else np.nan
            )
        else:
            modes = observed_target.mode()
            fallback = modes.iloc[0] if not modes.empty else np.nan

        (
            tree,
            predictors,
            _,
            encoder,
            numeric_fill_values,
        ) = _train_attribute_tree(
            complete_records,
            target_column,
        )

        if tree is None:
            result.loc[missing_indices, target_column] = fallback

            diagnostics["tree_fallbacks"].append(
                {
                    "attribute": target_column,
                    "reason": "No usable decision tree; used observed-value fallback.",
                }
            )
            continue

        # Route complete training records to their leaves.
        training_leaves = _assign_records_to_leaves(
            tree,
            predictors,
            complete_records,
            encoder,
            numeric_fill_values,
        )

        # Route records missing this target using the same fitted encoder.
        records_to_impute = original.loc[missing_indices].copy()

        missing_leaves = _assign_records_to_leaves(
            tree,
            predictors,
            records_to_impute,
            encoder,
            numeric_fill_values,
        )

        # Process each relevant leaf once for this target attribute.
        for leaf_id in missing_leaves.unique():
            target_indices = missing_leaves.index[
                missing_leaves == leaf_id
            ]

            leaf_training_indices = training_leaves.index[
                training_leaves == leaf_id
            ]

            if len(leaf_training_indices) == 0:
                result.loc[target_indices, target_column] = fallback
                continue

            if not is_numeric_target:
                # The tree predicts the categorical target class.
                leaf_values = complete_records.loc[
                    leaf_training_indices, target_column
                ].dropna()

                modes = leaf_values.mode()
                leaf_value = modes.iloc[0] if not modes.empty else fallback

                result.loc[target_indices, target_column] = leaf_value

            else:
                # Build a leaf-specific dataset containing the complete
                # training records and records missing this target.
                leaf_data = pd.concat(
                    [
                        original.loc[leaf_training_indices],
                        original.loc[target_indices],
                    ],
                    axis=0,
                )

                try:
                    leaf_imputed, emi_info = gaussian_emi(
                        leaf_data,
                        max_iter=max_iter,
                        tol=tol,
                        eigen_floor=eigen_floor,
                    )

                    diagnostics["emi_calls"] += 1
                    diagnostics["emi_diagnostics"].append(
                        {
                            "attribute": target_column,
                            "leaf_id": int(leaf_id),
                            "records": len(leaf_data),
                            "diagnostics": emi_info,
                        }
                    )

                    # Only copy the target's values for rows where that
                    # target was originally missing.
                    imputed_values = leaf_imputed.loc[
                        target_indices, target_column
                    ]

                    result.loc[
                        target_indices, target_column
                    ] = imputed_values

                except (ValueError, np.linalg.LinAlgError) as exc:
                    result.loc[target_indices, target_column] = fallback

                    diagnostics["tree_fallbacks"].append(
                        {
                            "attribute": target_column,
                            "leaf_id": int(leaf_id),
                            "reason": f"Gaussian EMI failed: {exc}",
                        }
                    )

    diagnostics["missing_cells_after"] = int(result.isna().sum().sum())

    return result, diagnostics

def evaluate_numeric_imputation(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict:
    """
    Evaluate numerical imputation using artificially hidden cells only.

    Returns per-column and pooled RMSE, R², and index of agreement (d2).
    The mask must be True where values were artificially hidden.
    """
    from sklearn.metrics import mean_squared_error, r2_score

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

        actual = original.loc[hidden, column].astype(float).to_numpy()
        predicted = imputed.loc[hidden, column].astype(float).to_numpy()

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(np.mean(np.abs(actual - predicted)))

        # Will be NaN when actual values have no variation.
        r2 = (
            float(r2_score(actual, predicted))
            if len(actual) >= 2 and np.var(actual) > 0
            else float("nan")
        )

        # Will be NaN if the index-of-agreement denominator is zero.
        denominator = np.sum(
            (
                np.abs(predicted - np.mean(actual))
                + np.abs(actual - np.mean(actual))
            ) ** 2
        )

        numerator = np.sum((predicted - actual) ** 2)

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
        return {"per_column": {}, "pooled": {}, "n_evaluated": 0}

    actual_all = np.asarray(all_actual, dtype=float)
    predicted_all = np.asarray(all_predicted, dtype=float)

    pooled_denominator = np.sum(
        (
            np.abs(predicted_all - np.mean(actual_all))
            + np.abs(actual_all - np.mean(actual_all))
        ) ** 2
    )
    pooled_numerator = np.sum((predicted_all - actual_all) ** 2)

    pooled = {
        "RMSE": float(
            np.sqrt(mean_squared_error(actual_all, predicted_all))
        ),
        "R2": (
            float(r2_score(actual_all, predicted_all))
            if len(actual_all) >= 2 and np.var(actual_all) > 0
            else float("nan")
        ),
        "d2": (
            float(1 - pooled_numerator / pooled_denominator)
            if pooled_denominator > 0
            else float("nan")
        ),
        "MAE": float(np.mean(np.abs(actual_all - predicted_all))),
    }

    return {
        "per_column": per_column,
        "pooled": pooled,
        "n_evaluated": len(actual_all),
    }
