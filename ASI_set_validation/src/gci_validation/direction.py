"""Paired normalized-HSIC residual direction scoring."""
from __future__ import annotations
import time
from dataclasses import replace
import numpy as np
from scipy.stats import normaltest, spearmanr
from .independence import hsic_test
from .regression import ResidualFit, out_of_fold_residuals
from .preprocessing import prepare_continuous_pair
from .result import DirectionResult

def _confidence_from_margin(score: float, scale: float) -> float:
    """Uncalibrated monotone reliability proxy; never reported as probability."""

    if not np.isfinite(score):
        return 0.0
    safe_scale = max(abs(scale), 1e-12)
    return float(np.clip(1.0 - np.exp(-abs(score) / safe_scale), 0.0, 1.0))

def anm_hsic_result(
    x: np.ndarray,
    y: np.ndarray,
    forward: ResidualFit,
    backward: ResidualFit,
    *,
    threshold: float,
    permutations: int,
    max_samples: int,
    seed: int,
) -> DirectionResult:
    start = time.perf_counter()
    test_forward = hsic_test(
        x,
        forward.residuals,
        permutations=permutations,
        max_samples=max_samples,
        seed=seed + 11,
    )
    test_backward = hsic_test(
        y,
        backward.residuals,
        permutations=permutations,
        max_samples=max_samples,
        # Reuse the same paired subsample/permutation stream in both directions.
        # This preserves the required score antisymmetry under X/Y swapping.
        seed=seed + 11,
    )
    score = test_backward.statistic - test_forward.statistic
    return DirectionResult(
        method="anm_hsic",
        score=score,
        direction=DirectionResult.direction_from_score(score, threshold),
        confidence=_confidence_from_margin(score, 0.02),
        applicable=True,
        diagnostics={
            "hsic_x_residual_y_on_x": test_forward.statistic,
            "hsic_y_residual_x_on_y": test_backward.statistic,
            "p_x_residual_y_on_x": test_forward.p_value,
            "p_y_residual_x_on_y": test_backward.p_value,
            "hsic_n": min(test_forward.n_used, test_backward.n_used),
            "hsic_permutations": permutations,
            "r2_y_on_x": forward.r2,
            "r2_x_on_y": backward.r2,
        },
        runtime_seconds=time.perf_counter() - start,
        seed=seed,
        calibrated=False,
    )

def analyze_direction(x: np.ndarray, y: np.ndarray, seed: int = 2026,
                      margin: float = 0.005) -> DirectionResult:
    """Return the fixed continuous-pair ANM-HSIC decision and diagnostics.

    Positive scores support x -> y. A near-linear-Gaussian or weak-association
    pair is left unresolved. The margin is not a calibrated probability.
    """
    try:
        pair = prepare_continuous_pair(x, y, min_samples=40, max_samples=5000, seed=seed)
    except ValueError as exc:
        return DirectionResult.invalid('anm_hsic', str(exc), seed=seed)
    forward = out_of_fold_residuals(pair.x, pair.y, n_splits=5, seed=seed)
    backward = out_of_fold_residuals(pair.y, pair.x, n_splits=5, seed=seed)
    linear_forward = out_of_fold_residuals(pair.x, pair.y, regressor='linear', n_splits=5, seed=seed)
    linear_backward = out_of_fold_residuals(pair.y, pair.x, regressor='linear', n_splits=5, seed=seed)
    rho = float(spearmanr(pair.x, pair.y).statistic)
    raw = hsic_test(pair.x, pair.y, permutations=0, max_samples=500, seed=seed+7)
    association = bool(raw.statistic >= 0.01 or abs(rho) >= 0.10)
    normality = [float(normaltest(v).pvalue) for v in
                 [pair.x, pair.y, linear_forward.residuals, linear_backward.residuals]]
    nonlinear_gain = max(forward.r2-linear_forward.r2, backward.r2-linear_backward.r2)
    linear_gaussian = bool(association and nonlinear_gain <= 0.03 and min(normality) > 0.05)
    result = anm_hsic_result(pair.x, pair.y, forward, backward, threshold=margin,
                             permutations=0, max_samples=500, seed=seed)
    if not association or linear_gaussian:
        diagnostics = dict(result.diagnostics)
        diagnostics['direction_gate_reason'] = ('no_detectable_pair_dependence' if not association
                                                 else 'linear_gaussian_direction_nonidentifiable')
        result = replace(result, direction='undirected', diagnostics=diagnostics)
    return result
