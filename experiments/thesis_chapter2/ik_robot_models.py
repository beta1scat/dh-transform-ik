"""Robot and method selection for Chapter 2."""
from experiments.common.rtb_models import build_rtb_robot

ROBOT_FILES = {
    "abb_irb2600_20_165": "abb_irb2600_20_165.json",
    "estun_er8_720mi": "estun_er8_720mi.json",
    "kuka_kr50_r2500": "kuka_kr50_r2500.json",
    "yaskawa_gp12": "yaskawa_gp12.json",
}
METHODS = ("ikine_min", "ikine_LM", "ikine_LMS")
T_FIELDS = [f"target_{row}{column}" for row in range(4) for column in range(4)]


