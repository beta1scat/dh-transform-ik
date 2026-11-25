"""FK regression checks for singular reference mapping and coupled limits."""

import math
from pathlib import Path
import unittest

import numpy as np

from dh_transform_ik import load_robot, solve_ik, transform_dh
from dh_transform_ik.ik import _closest_coupled_wrist
from experiments.common import forward_kinematics
from experiments.thesis_chapter2.ik_comparison_protocol import evaluate_solution, joint_limits_rad


ROOT = Path(__file__).resolve().parents[1]


class SingularReferenceTests(unittest.TestCase):
    def test_coupled_projection_obeys_limits_and_both_reference_angles(self):
        # q4 + q6 = 1; unconstrained projection is (0.6, 0.4).
        pair = _closest_coupled_wrist(
            0.0, 1.0, -1.0, (-1.0, -1.0), (1.0, 1.0), (0.8, 0.6)
        )
        np.testing.assert_allclose(pair, (0.6, 0.4), atol=1e-12)
        # A tight q4 limit requires a different feasible allocation.
        pair = _closest_coupled_wrist(
            0.0, 1.0, -1.0, (-0.2, -1.0), (0.2, 1.0), (0.8, 0.6)
        )
        np.testing.assert_allclose(pair, (0.2, 0.8), atol=1e-12)

    def test_reference_mapping_and_singular_roundtrip(self):
        template = load_robot(str(ROOT / "robots/fanuc_lr_mate_200id.json"))
        for filename in (
            "fanuc_lr_mate_200id.json", "kuka_kr50_r2500.json",
            "abb_irb2600_20_165.json", "yaskawa_gp12.json", "estun_er8_720mi.json",
        ):
            robot = load_robot(str(ROOT / "robots" / filename))
            transformed = transform_dh(template, robot)
            lower, upper = joint_limits_rad(robot)
            source = (lower + upper) / 2.0
            # Choose an actual joint angle corresponding to template theta5=0.
            dh5 = transformed[5]
            direction = 1.0 if dh5.is_co_orient else -1.0
            source[4] = direction * math.radians(dh5.zero_offset)
            source[3] = lower[3] + 0.6 * (upper[3] - lower[3])
            source[5] = lower[5] + 0.4 * (upper[5] - lower[5])
            target = forward_kinematics(robot, np.degrees(source))
            with self.subTest(robot=filename):
                solutions = solve_ik(
                    target, template, robot, transformed_dh=transformed,
                    reference_q=source,
                )
                accepted = []
                for q in solutions.values():
                    checked = evaluate_solution(q, target, robot, lower, upper, 1e-3)
                    if checked["experiment_success"]:
                        accepted.append(np.array([checked[f"q{i}_rad"] for i in range(1, 7)]))
                self.assertTrue(accepted, "No singular solution passed FK and limit checks")
                wrist_distance = min(np.linalg.norm(q[[3, 5]] - source[[3, 5]]) for q in accepted)
                self.assertLess(wrist_distance, 1e-5)


if __name__ == "__main__":
    unittest.main()
