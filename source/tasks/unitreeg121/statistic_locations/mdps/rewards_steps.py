from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.mdps.rewards import rew_statistics

@configclass
class RewardsStepsCfg():

    rew_feetpose_symmetry = RewardTermCfg(
        func=rew_statistics.rew_feetpose_symmetry,
        weight= 0.15,
        params={"asset_cfg": SceneEntityCfg("robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link"
                        ]),
                "statistics_name": "steps",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.04,
                }
    )
    rew_bodiespose_symmetry = RewardTermCfg(
        func=rew_statistics.rew_bodiespose_symmetry,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    body_names=[
                            "left_elbow_link",
                            "right_elbow_link",
                            "left_rubber_hand",
                            "right_rubber_hand",
                        ]),
                "statistics_name": "steps",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.04,
                }
    )
    rew_jointspose_symmetry = RewardTermCfg(
        func=rew_statistics.rew_jointspose_symmetry,
        weight= 0.1,
        params={"asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint", "right_hip_pitch_joint",
                        "left_knee_joint", "right_knee_joint",
                        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                        "left_elbow_joint", "right_elbow_joint",
                        ]),
                "statistics_name": "steps",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.1,
                }
    )
    rew_times_symmetry = RewardTermCfg(
        func=rew_statistics.rew_times_symmetry,
        weight= 0.1,
        params={
                "statistics_name": "steps",
                "type": rew_statistics.mirror_or_synchronize.MIRROR,
                "error_std": 0.06,
                }
    )

