"""Probe nonzero d-offset propagation with controlled synthetic MDH variants."""

from __future__ import annotations

import argparse
import copy
import random

import numpy as np

try:
    from common import (
        DEFAULT_SEED,
        PROCESSED_DIR,
        TEMPLATE_ROBOT,
        forward_kinematics,
        map_arbitrary_to_standard_deg,
        pose_errors,
        sample_joint_angles,
        transformed_forward_kinematics,
        write_csv,
    )
except ImportError:
    from experiments.common import (
        DEFAULT_SEED,
        PROCESSED_DIR,
        TEMPLATE_ROBOT,
        forward_kinematics,
        map_arbitrary_to_standard_deg,
        pose_errors,
        sample_joint_angles,
        transformed_forward_kinematics,
        write_csv,
    )

from dh_transform_ik import load_robot, transform_dh


CASES = (
    ("joint_2_d_plus_25", 1, 25.0),
    ("joint_6_d_plus_25", 5, 25.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    template = load_robot(str(TEMPLATE_ROBOT.source_path))
    rows = []
    summaries = []
    for case_name, index, delta_mm in CASES:
        arbitrary = copy.deepcopy(template)
        arbitrary[index].d += delta_mm
        transformed = transform_dh(template, arbitrary)
        case_errors = []
        for trial in range(args.trials):
            q_arbitrary = sample_joint_angles(arbitrary, rng)
            q_standard = map_arbitrary_to_standard_deg(q_arbitrary, transformed)
            original = forward_kinematics(arbitrary, q_arbitrary)
            mapped = transformed_forward_kinematics(transformed, q_standard)
            errors = pose_errors(original, mapped)
            case_errors.append(errors.matrix_max_abs_error)
            rows.append(
                {
                    "case": case_name,
                    "modified_index": index,
                    "delta_d_mm": delta_mm,
                    "trial": trial,
                    **errors.to_dict(),
                }
            )
        values = np.asarray(case_errors, dtype=np.float64)
        summaries.append(
            {
                "case": case_name,
                "modified_index": index,
                "delta_d_mm": delta_mm,
                "trials": args.trials,
                "mean_matrix_max_abs_error": float(np.mean(values)),
                "max_matrix_max_abs_error": float(np.max(values)),
                "fk_equivalent_at_1e_10": bool(np.all(values <= 1e-10)),
            }
        )

    write_csv(
        PROCESSED_DIR / "d_offset_validation.csv",
        [
            "case",
            "modified_index",
            "delta_d_mm",
            "trial",
            "position_error_mm",
            "rotation_error_deg",
            "matrix_mean_abs_error",
            "matrix_max_abs_error",
        ],
        rows,
    )
    write_csv(
        PROCESSED_DIR / "d_offset_validation_summary.csv",
        list(summaries[0]),
        summaries,
    )


if __name__ == "__main__":
    main()
