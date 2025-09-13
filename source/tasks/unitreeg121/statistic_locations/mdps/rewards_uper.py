from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.mdps.rewards import rew_statistics


@configclass
class RewardsUperCfg():
    # shoulderp
    rew_mean_uper_step_symmetry = RewardTermCfg(
        func=rew_statistics.rew_mean_step_symmetry,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "left_elbow_joint",
                        "right_elbow_joint"
                        ]),
                "pos_statistics_name": "pos",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.08,
                }
    )
    rew_var_uper_symmetry = RewardTermCfg(
        func=rew_statistics.rew_variance_symmetry,
        weight=0.1,
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
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_.*_joint",
                        ".*_elbow_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.15,

                }
    )
    rew_var_uper_zero = RewardTermCfg(
        func=rew_statistics.rew_variance_zero,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        ".*_shoulder_.*_joint",
                        ".*_elbow_joint",
                        "waist.*",
                        ]),
                "pos_statistics_name": "pos",
                "error_std": 0.005,
                }
    )
