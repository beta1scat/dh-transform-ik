"""Metre-based Robotics Toolbox model corresponding to the original MDH chain."""
import math
import roboticstoolbox as rtb
from spatialmath import SE3
from dh_transform_ik import forward_kinematics_mdh


def build_rtb_robot(dh_list):
    links = []
    for dh in dh_list[:6]:
        links.append(
            rtb.RevoluteMDH(
                alpha=math.radians(dh.alpha),
                a=dh.a / 1000.0,
                offset=math.radians(dh.theta_offset),
                d=dh.d / 1000.0,
            )
        )
    end = dh_list[6]
    tool_mm = forward_kinematics_mdh(
        end.alpha, end.a, end.theta_offset, end.d, 0.0
    )
    tool_m = tool_mm.copy()
    tool_m[:3, 3] /= 1000.0
    return rtb.DHRobot(links, tool=SE3(tool_m), name="ik_comparison_model")
