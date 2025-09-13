from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.mdps.rewards import rew_statistics

@configclass
class RewardsLegCfg():
    # hipp knee
    rew_mean_hk_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_symmetry,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.1,
                }
    )
    rew_mean_hk_step_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_step_symmetry,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.08,
                }
    )
    rew_var_hk_symmetry = RewardTermCfg(
        func=rew_statistics.rew_variance_symmetry,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.008,
                }
    )
    rew_var_hk_constraint = RewardTermCfg(
        func=rew_statistics.rew_variance_constraint,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",

                "max_constraint": 0.125,
                "min_constraint": 0.045,
                "error_std": 0.015,
                }
    )
    rew_mean_leg_zero = RewardTermCfg(
        func=rew_statistics.rew_mean_zero,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_hip_roll_joint",
                        ".*_hip_yaw_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.15,
                }
    )
    rew_var_leg_zero = RewardTermCfg(
        func=rew_statistics.rew_variance_zero,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_hip_roll_joint",
                        ".*_hip_yaw_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.005,
                }
    )