from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.mdps.rewards import rew_task, rew_actions, rew_joints, rew_bodies, rew_feet, rew_episode
import math

@configclass
class RewardsCfg:
    # task
    rew_lin_xy_exp = RewardTermCfg(
        func=rew_task.rew_lin_xy_exp2,
        weight=1,
        params={"std": math.sqrt(0.25),
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_ang_z_exp = RewardTermCfg(
        func=rew_task.rew_ang_z_exp,
        weight=1,
        params={"std": math.sqrt(0.25),
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    rew_motion_speed = RewardTermCfg(
        func=rew_task.rew_motion_speed,
        weight=0.75,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_motion_hard = RewardTermCfg(
        func=rew_task.rew_motion_hard,
        weight=0.6,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )

    p_linear_velocity = RewardTermCfg(func=rew_task.p_lin_z_l2, weight=-2.0)
    p_angular_velocity = RewardTermCfg(func=rew_task.p_ang_xy_l2, weight=-0.05)

    # action # 4
    p_action_rate = RewardTermCfg(
        func=rew_actions.p_action_rate_l2, weight=-0.02)

    p_action_smoothness = RewardTermCfg(
        func=rew_actions.p_action_smoothness,
        weight=-0.005,
        params={
            "weight1": 1,
            "weight2": 1,
            "weight3": 0.05,
            },
    )

    # joints
    p_energy = RewardTermCfg(
        func=rew_joints.p_energy,
        weight=-2e-5,
        params={
            "asset_cfg":
                SceneEntityCfg("robot")
            },
    )

    p_pos_limits = RewardTermCfg(
        func=rew_joints.p_jpos_limits_l1,
        weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_jvel = RewardTermCfg(func=rew_joints.p_jvel_l2, weight=-0.001)
    p_jacc = RewardTermCfg(func=rew_joints.p_jacc_l2, weight=-3e-8)

    p_deviation_arms = RewardTermCfg(
        func=rew_joints.p_jpos_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    # ".*_wrist_.*",
                ],
            )
        },
    )
    p_deviation_waists = RewardTermCfg(
        func=rew_joints.p_jpos_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    p_deviation_legs = RewardTermCfg(
        func=rew_joints.p_jpos_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # body
    p_legwidth = RewardTermCfg(
        func=rew_bodies.p_width,
        weight=-5,
        params={
            "target_width": 0.238,  # Adjusting for the foot clearance
            "target_height": 0.78,
            "center_velocity": 1.8,
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link",
                                     ".*left_knee_link",
                                     ".*right_knee_link"])
            },
    )
    p_handwidth = RewardTermCfg(
        func=rew_bodies.p_width,
        weight=-10,
        params={
            "target_width": 0.30,  # Adjusting for the foot clearance
            "target_height": 0.78,
            "center_velocity": 1.8,
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=[".*left_elbow_link",
                                     ".*right_elbow_link",
                                     ".*left_rubber_hand",
                                     ".*right_rubber_hand"])
            },
    )
    p_orientation = RewardTermCfg(
        func=rew_bodies.p_ori_l2,
        weight=-5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_height = RewardTermCfg(
        func=rew_bodies.p_height_base2feet,
        weight=-8.0, params={
            "target_height": 0.78,  # Adjusting for the foot clearance
            "asset_cfg": SceneEntityCfg("robot",
                         body_names=".*_ankle_roll_link")}
    )
    rp_height_upper = RewardTermCfg(
        func=rew_bodies.rp_height_upper,
        weight=0.25, params={
            "target_height": 0.46018,  # Adjusting for the foot clearance
            "error_std": 0.06,
            "penalize_weight": -0.25,
            "asset_cfg": SceneEntityCfg("robot",
                         body_names="mid360_link")}
    )

    p_uncontacts = RewardTermCfg(
        func=rew_bodies.p_undesired_contacts,
        weight=-0.15,
        params={
            'threshold': 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                                'pelvis',
                                'imu_in_pelvis',
                                'left_hip_pitch_link',
                                'left_hip_roll_link',
                                'left_hip_yaw_link',
                                'left_knee_link',
                                'pelvis_contour_link',
                                'right_hip_pitch_link',
                                'right_hip_roll_link',
                                'right_hip_yaw_link',
                                'right_knee_link',
                                'torso_link',
                                'd435_link',
                                'head_link',
                                'imu_in_torso',
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'left_rubber_hand',
                                'logo_link',
                                'mid360_link',
                                'right_shoulder_pitch_link',
                                'right_shoulder_roll_link',
                                'right_shoulder_yaw_link',
                                'right_elbow_link',
                                'right_rubber_hand'
            ])
        }
    )

    rew_stability= RewardTermCfg(
        func=rew_bodies.rew_stability,
        weight=0.2,
        params={"asset_cfg":
                SceneEntityCfg("robot", body_names=[
                                     ".*left_ankle_roll_link",
                                     ".*right_ankle_roll_link"])},
    )
    # feet
    rew_feet_air_time = RewardTermCfg(
        func=rew_feet.rew_air_time_biped,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    p_feet_vertical_force = RewardTermCfg(
        func=rew_feet.p_forces_z,
        weight=-5e-3,
        params={
            "threshold": 500,
            "max_forces": 400,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    p_feet_stumble = RewardTermCfg(
        func=rew_feet.p_stumble,
        weight=-1e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
        },
    )
    p_feet_slide = RewardTermCfg(
        func=rew_feet.p_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        },
    )
    p_feet_in_air = RewardTermCfg(
        func=rew_feet.p_both_feet_in_air,
        weight=-0.02,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}
    )
    p_foot_clearance = RewardTermCfg(
        func=rew_feet.p_max_feet_height_before_contact,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            'target_height': 0.16 + 0.055
        },
    )
    p_feet_ori = RewardTermCfg(
        func=rew_feet.p_feet_orientation,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names='.*_ankle_roll_link'),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            }
    )

    # -------------------- Episode Penalty --------------------
    p_termination = RewardTermCfg(
        func=rew_episode.p_eps_terminated,
        weight=-100,
    )

if False:
    from .rewards_uper import RewardsUperCfg
    from .rewards_leg import RewardsLegCfg
else:
    from .rewards_uper import RewardsUperCfg
    from .rewards_leg import RewardsLegCfg

@configclass
class RewardsG21Cfg(RewardsUperCfg, RewardsLegCfg, RewardsCfg):
    pass
