"""Shared acceptance rules for the chapter 2 IK comparison."""

from __future__ import annotations

import math
import numpy as np

from experiments.common import forward_kinematics, pose_errors


def joint_limits_rad(robot_dh) -> tuple[np.ndarray, np.ndarray]:
    lower = np.radians([link.theta_min for link in robot_dh[:6]])
    upper = np.radians([link.theta_max for link in robot_dh[:6]])
    if not np.all(np.isfinite([lower, upper])) or np.any(lower > upper):
        raise ValueError("Invalid robot joint limits")
    return lower, upper


def periodic_solution_in_limits(q, lower, upper):
    """Choose a 2*pi-equivalent value in each interval; never clip an angle."""
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        return None
    tolerance = 1e-9
    result = q.copy()
    for index, value in enumerate(q):
        if lower[index] - tolerance <= value <= upper[index] + tolerance:
            continue
        first = math.ceil((lower[index] - value - tolerance) / (2.0 * math.pi))
        last = math.floor((upper[index] - value + tolerance) / (2.0 * math.pi))
        if first > last:
            return None
        nearest = round(((lower[index] + upper[index]) / 2.0 - value) / (2.0 * math.pi))
        result[index] = value + min(max(nearest, first), last) * 2.0 * math.pi
    return result


def evaluate_solution(q, target_mm, robot_dh, lower, upper, me_tolerance):
    """Validate both solver families using the original MDH chain in mm."""
    q = np.asarray(q, dtype=float)
    finite = q.shape == (6,) and bool(np.all(np.isfinite(q)))
    normalized = periodic_solution_in_limits(q, lower, upper) if finite else None
    result = {
        "finite": finite,
        "within_limits": normalized is not None,
        "experiment_success": False,
        "matrix_mean_error": math.nan,
        "position_error_mm": math.nan,
        "rotation_error_deg": math.nan,
        "failure_reason": "nonfinite_or_invalid_shape" if not finite else "joint_limits",
    }
    if finite:
        checked_q = normalized if normalized is not None else q
        actual = forward_kinematics(robot_dh, np.degrees(checked_q))
        metrics = pose_errors(target_mm, actual)
        result.update(
            matrix_mean_error=metrics.matrix_mean_abs_error,
            position_error_mm=metrics.position_error_mm,
            rotation_error_deg=metrics.rotation_error_deg,
        )
        passed = math.isfinite(metrics.matrix_mean_abs_error) and metrics.matrix_mean_abs_error <= me_tolerance
        result["experiment_success"] = normalized is not None and passed
        if normalized is not None:
            result["failure_reason"] = "" if passed else "pose_tolerance"
    output_q = normalized if normalized is not None else (q if finite else np.full(6, math.nan))
    result.update({f"q{i + 1}_rad": float(value) for i, value in enumerate(output_q)})
    return result


def initial_guesses(seed: int, robot_index: int, trial: int, count: int, lower, upper):
    """Identical midpoint/random starts for every numerical method on a target."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, robot_index, trial]))
    guesses = [(lower + upper) / 2.0]
    guesses.extend(rng.uniform(lower, upper, size=(count - 1, 6)))
    return guesses
