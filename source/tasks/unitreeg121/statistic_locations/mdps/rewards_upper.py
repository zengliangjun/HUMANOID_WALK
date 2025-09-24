from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.mdps.rewards import rew_statistics


@configclass
class RewardsUperCfg():
    # shoulderp
    rew_mean_uper_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_symmetry,
        weight= 0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.085,
                }
    )
    rew_var_uper_symmetry = RewardTermCfg(
        func=rew_statistics.rew_variance_symmetry,
        weight=0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.085,
                }
    )
    rew_mean_uper_zero = RewardTermCfg(
        func=rew_statistics.rew_mean_zero,
        weight=0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.045,

                }
    )
    rew_var_uper_zero = RewardTermCfg(
        func=rew_statistics.rew_variance_zero,
        weight=0.015,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.045,
                }
    )


@configclass
class PBRSUperCfg(RewardsUperCfg):
    # shoulderp
    rew_mean_uper_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_mean_symmetry,
        weight= 1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.085,

                "sigma": 0.45,
                }
    )
    rew_var_uper_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_variance_symmetry,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.085,

                "sigma": 0.65,
                }
    )
    rew_mean_uper_zero = RewardTermCfg(
        func=rew_statistics.pbrs_mean_zero,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.045,

                "sigma": 1.15,

                }
    )
    rew_var_uper_zero = RewardTermCfg(
        func=rew_statistics.pbrs_variance_zero,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.045,

                "sigma": 0.85,
                }
    )

    rew_bodies_uper_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_bodies_symmetry,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                ],
                preserve_order=True
            ),
            "command_name": "base_velocity",
            "error_std": 0.06,

            "sigma": 0.85,
        },
    )

