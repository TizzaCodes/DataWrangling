"""
Gaussian EM imputation for numerical columns.

This is a transparent baseline implementation of the numerical EMI component.
It does not impute categorical columns; those are intentionally left unchanged
for the paper-specific method/evaluation decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_psd(matrix: np.ndarray, eigen_floor: float = 1e-8) -> np.ndarray:
    """Symmetrize a covariance matrix and floor its eigenvalues."""
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eigen_floor)
    return (vectors * values) @ vectors.T


def _conditional_moments(
    observed_values: np.ndarray,
    observed_idx: np.ndarray,
    missing_idx: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional expected full rows and covariance of missing block."""
    n_rows = observed_values.shape[0]
    n_features = len(mean)

    expected = np.empty((n_rows, n_features), dtype=float)
    conditional_cov = np.zeros((n_features, n_features), dtype=float)

    if len(observed_idx):
        expected[:, observed_idx] = observed_values
    if len(missing_idx) == 0:
        return expected, conditional_cov

    if len(observed_idx) == 0:
        expected[:, :] = mean
        conditional_cov[np.ix_(missing_idx, missing_idx)] = covariance[
            np.ix_(missing_idx, missing_idx)
        ]
        return expected, conditional_cov

    cov_oo = covariance[np.ix_(observed_idx, observed_idx)]
    cov_mo = covariance[np.ix_(missing_idx, observed_idx)]
    cov_om = covariance[np.ix_(observed_idx, missing_idx)]
    cov_mm = covariance[np.ix_(missing_idx, missing_idx)]

    # Solve rather than explicitly invert for better numerical stability.
    solve_residuals = np.linalg.solve(
        cov_oo, (observed_values - mean[observed_idx]).T
    ).T
    conditional_mean = mean[missing_idx] + solve_residuals @ cov_mo.T
    expected[:, missing_idx] = conditional_mean

    solved_cov = np.linalg.solve(cov_oo, cov_om)
    cond_missing_cov = cov_mm - cov_mo @ solved_cov
    cond_missing_cov = (cond_missing_cov + cond_missing_cov.T) / 2.0
    conditional_cov[np.ix_(missing_idx, missing_idx)] = cond_missing_cov

    return expected, conditional_cov


def gaussian_emi(
    data: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-5,
    eigen_floor: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing numerical values using Gaussian EM.

    Parameters
    ----------
    data:
        DataFrame with a mix of numerical and non-numerical columns.
    max_iter:
        Maximum EM iterations.
    tol:
        Convergence threshold for the standardized mean/covariance updates.
    eigen_floor:
        Minimum covariance eigenvalue after regularization.

    Returns
    -------
    imputed:
        Copy of data with numerical missing values imputed. Categorical
        missing values are not modified.
    diagnostics:
        Iteration count, convergence status, numeric columns, and missing-cell
        counts before/after.
    """
    imputed = data.copy()
    numeric_columns = list(data.select_dtypes(include=[np.number]).columns)

    if not numeric_columns:
        raise ValueError("No numerical columns found for Gaussian EMI.")

    numeric = data[numeric_columns].astype(float)
    raw = numeric.to_numpy(copy=True)
    missing = np.isnan(raw)
    missing_before = int(missing.sum())

    if missing_before == 0:
        return imputed, {
            "numeric_columns": numeric_columns,
            "numeric_missing_before": 0,
            "numeric_missing_after": 0,
            "iterations": 0,
            "converged": True,
            "note": "No missing numerical values in this dataset.",
        }

    # Scale each feature using observed values to avoid covariance domination
    # by columns measured on large scales.
    centers = np.nanmean(raw, axis=0)
    scales = np.nanstd(raw, axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
    z = (raw - centers) / scales

    n_rows, n_features = z.shape
    missing = np.isnan(z)
    patterns, inverse = np.unique(missing, axis=0, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(patterns))]

    # Mean-filled covariance is an initialization only; EM performs the updates.
    initial = z.copy()
    initial_means = np.nanmean(initial, axis=0)
    rows, cols = np.where(np.isnan(initial))
    initial[rows, cols] = initial_means[cols]
    mean = np.mean(initial, axis=0)
    covariance = np.cov(initial, rowvar=False, ddof=0)
    if n_features == 1:
        covariance = np.array([[float(covariance)]])
    covariance = _nearest_psd(np.atleast_2d(covariance), eigen_floor)

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        sum_x = np.zeros(n_features, dtype=float)
        sum_xx = np.zeros((n_features, n_features), dtype=float)

        for pattern_idx, row_indices in enumerate(groups):
            pattern = patterns[pattern_idx]
            missing_idx = np.flatnonzero(pattern)
            observed_idx = np.flatnonzero(~pattern)
            observed_values = z[np.ix_(row_indices, observed_idx)]

            expected, conditional_cov = _conditional_moments(
                observed_values,
                observed_idx,
                missing_idx,
                mean,
                covariance,
            )
            sum_x += expected.sum(axis=0)
            sum_xx += expected.T @ expected
            sum_xx += len(row_indices) * conditional_cov

        new_mean = sum_x / n_rows
        new_covariance = sum_xx / n_rows - np.outer(new_mean, new_mean)
        new_covariance = _nearest_psd(new_covariance, eigen_floor)

        delta = max(
            float(np.max(np.abs(new_mean - mean))),
            float(np.max(np.abs(new_covariance - covariance))),
        )
        mean, covariance = new_mean, new_covariance

        if delta < tol:
            converged = True
            break

    # Conditional-mean imputation using the final estimated parameters.
    completed_z = z.copy()
    for pattern_idx, row_indices in enumerate(groups):
        pattern = patterns[pattern_idx]
        missing_idx = np.flatnonzero(pattern)
        if len(missing_idx) == 0:
            continue
        observed_idx = np.flatnonzero(~pattern)
        observed_values = z[np.ix_(row_indices, observed_idx)]
        expected, _ = _conditional_moments(
            observed_values,
            observed_idx,
            missing_idx,
            mean,
            covariance,
        )
        completed_z[np.ix_(row_indices, missing_idx)] = expected[
            :, missing_idx
        ]

    completed_raw = completed_z * scales + centers
    imputed.loc[:, numeric_columns] = completed_raw
    numeric_after = int(imputed[numeric_columns].isna().sum().sum())

    return imputed, {
        "numeric_columns": numeric_columns,
        "numeric_missing_before": missing_before,
        "numeric_missing_after": numeric_after,
        "iterations": iteration,
        "converged": converged,
        "tolerance": tol,
        "max_iterations": max_iter,
        "unique_numeric_missing_patterns": len(patterns),
        "note": (
            "Numerical columns imputed by Gaussian EM; categorical missing "
            "values intentionally remain missing."
        ),
    }
