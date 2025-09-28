from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.mdps.rewards import rew_statistics

@configclass
class RewardsCfg():
    rew_symmetry_joints_leg = RewardTermCfg(
        func=rew_statistics.rew_joints_symmetry,
        weight= 0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        ]),
            "with_default_or_zero": 0, # 0 is zero, 1 is default
            "error_std": 0.12,
            "penalize_weight": -0.25,
            "min_constraint": 0.195, # 0.045
            "max_constraint": 0.245,
            "ratios": [0.45, 0.45, 0.1]
        }
    )
    rew_symmetry_joints_uper = RewardTermCfg(
        func=rew_statistics.rew_joints_symmetry,
        weight= 0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "right_shoulder_pitch_joint",
                        ]),
            "with_default_or_zero": 1, # 0 is zero, 1 is default
            "error_std": 0.12,
            "penalize_weight": -0.25,
            "min_constraint": 0.1, # 0.045
            "max_constraint": 0.245,
            "ratios": [0.45, 0.45, 0.1]
        }
    )
    rew_zero_joints_leg = RewardTermCfg(
        func=rew_statistics.rew_joints_zero,
        weight= 0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                            ".*_hip_roll_joint",
                            ".*_hip_yaw_joint",
                            ".*_ankle_roll_joint",
                        ]),
            "with_default_or_zero": 1, # 0 is zero, 1 is default
            "error_std": 0.12,
            "penalize_weight": -0.25,
            "ratios": [0.5, 0.5]
        }
    )
    rew_zero_joints_uper = RewardTermCfg(
        func=rew_statistics.rew_joints_zero,
        weight= 0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[
                            ".*_shoulder_roll_joint",
                            ".*_shoulder_yaw_joint",
                            "waist.*",
                        ]),
            "with_default_or_zero": 1, # 0 is zero, 1 is default
            "error_std": 0.12,
            "penalize_weight": -0.25,
            "ratios": [0.5, 0.5]
        }
    )
    rew_symmetry_bodies_leg = RewardTermCfg(
        func=rew_statistics.rew_bodies_symmetry,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True
            ),
            "command_name": "base_velocity",
            "error_std": 0.04,
        },
    )
    rew_symmetry_bodies_uper = RewardTermCfg(
        func=rew_statistics.rew_bodies_symmetry,
        weight=0.1,
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
        },
    )

    rew_symmetry_linvel_leg = RewardTermCfg(
        func=rew_statistics.rew_linvel_symmetry,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True
            ),
            "command_name": "base_velocity",
            "error_std": 0.3,
            "penalize_weight": -0.25
        },
    )

    rew_symmetry_linvel_uper = RewardTermCfg(
        func=rew_statistics.rew_linvel_symmetry,
        weight=0.1,
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
            "error_std": 0.3,
            "penalize_weight": -0.25
        },
    )

