from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.mdps.rewards import rew_statistics


@configclass
class RewardsUperCfg():
    # shoulderp
    rew_mean_uper_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_symmetry,
        weight= 0.04,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.05,
                }
    )
    rew_var_uper_symmetry = RewardTermCfg(
        func=rew_statistics.rew_variance_symmetry,
        weight=0.04,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.008,
                }
    )
    rew_mean_uper_zero = RewardTermCfg(
        func=rew_statistics.rew_mean_zero,
        weight=0.04,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.02,

                }
    )
    rew_var_uper_zero = RewardTermCfg(
        func=rew_statistics.rew_variance_zero,
        weight=0.04,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.005,
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
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.05,
                }
    )
    rew_var_uper_symmetry = RewardTermCfg(
        func=rew_statistics.pbrs_variance_symmetry,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.008,
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
                "error_std": 0.02,

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
                "error_std": 0.005,
                }
    )
