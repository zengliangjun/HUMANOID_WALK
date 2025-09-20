from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.mdps.rewards import rew_statistics, rew_feet

@configclass
class RewardsLegCfg():
    # hipp knee
    rew_mean_leg_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_symmetry,
        weight= 0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.1,
                }
    )
    '''
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
    '''
    rew_var_leg_symmetry = RewardTermCfg(
        func=rew_statistics.rew_variance_symmetry,
        weight=0.015,
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

    rew_var_leg_constraint = RewardTermCfg(
        func=rew_statistics.rew_variance_constraint,
        weight=0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",

                "min_constraint": 0.035, # 0.045
                "max_constraint": 0.06125,
                "error_std": 0.008,
                }
    )

    rew_mean_leg_zero = RewardTermCfg(
        func=rew_statistics.rew_mean_zero,
        weight=0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_hip_roll_joint",
                        ".*_hip_yaw_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.02,
                }
    )
    rew_var_leg_zero = RewardTermCfg(
        func=rew_statistics.rew_variance_zero,
        weight=0.015,
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

@configclass
class PBRSLegCfg(RewardsLegCfg):
    rew_mean_leg_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_mean_symmetry,
        weight= 1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.1,

                "sigma": 0.85,
                }
    )
    rew_var_leg_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_variance_symmetry,
        weight=0.5,
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

                "sigma": 0.85,
                }
    )
    rew_var_leg_constraint = RewardTermCfg(
        func=rew_statistics.pbrs_variance_constraint,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "left_knee_joint",
                        "right_knee_joint"
                        ]),
                "pos_statistics_name": "pos",

                "min_constraint": 0.035, # 0.045
                "max_constraint": 0.06125,
                "error_std": 0.004,

                "sigma": 0.85,
                }
    )
    rew_mean_leg_zero = RewardTermCfg(
        func=rew_statistics.pbrs_mean_zero,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_hip_roll_joint",
                        ".*_hip_yaw_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.02,

                "sigma": 0.85,
                }
    )
    rew_var_leg_zero = RewardTermCfg(
        func=rew_statistics.pbrs_variance_zero,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_hip_roll_joint",
                        ".*_hip_yaw_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.005,

                "sigma": 0.85,
                }
    )

    rew_bodies_symmetry = RewardTermCfg(
        func=rew_statistics.rew_bodies_symmetry,
        weight=0.1,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ]),

                "bodies_statistics_name": "bodies",
                "std_ranges": [0.14, 0.21],
                "error_std": 0.04,
                }
    )

    rew_times_symmetry = RewardTermCfg(
        func=rew_feet.rew_times_symmetry,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*left_ankle_roll_link",
                    ".*right_ankle_roll_link"
                ]
            ),
            "error_std": 0.06
        },
    )
