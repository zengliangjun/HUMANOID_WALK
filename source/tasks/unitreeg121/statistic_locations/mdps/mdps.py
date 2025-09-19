from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


from isaaclabex.mdps.commands import commands_cfg
from isaaclabex.mdps.statistics import joints, bodies
from isaaclabex.envs.managers import term_cfg

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "step_joint_names": [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint",
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_elbow_joint", "right_elbow_joint",

                "left_hip_roll_joint",  "right_hip_roll_joint",
                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                "left_ankle_roll_joint","right_ankle_roll_joint",
                "left_ankle_pitch_joint","right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            ]},

        # episode_truncation = 80,
        export_interval = 1000000
    )
    bodies = term_cfg.StatisticsTermCfg(
        func= bodies.StatusPose3d,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_elbow_link",
                    "right_elbow_link",
                    "left_rubber_hand",
                    "right_rubber_hand",
                ],
                preserve_order = True)},

        # episode_truncation = 80,
        export_interval = 1000000
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            #lin_vel_x=(0, 4.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-2., 2.), heading=(0., 0)
            lin_vel_x=(0, 2.8), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-2., 2.), heading=(0., 0)
        ),
        small2zero_threshold_line=0.25,
        small2zero_threshold_angle=0.25
    )

    def __post_init__(self):
        self.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 3.14 * 45 / 180})

    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})
