from isaaclab.utils import configclass
from isaaclabex.mdps.commands import commands_cfg
from isaaclabex.mdps.commands.zero2small_command import SymmetryCommand

from isaaclabex.mdps.observations import command_extends
from isaaclab.managers import ObservationTermCfg

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        class_type = SymmetryCommand,
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
             lin_vel_x=(0, 0.3), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.05, 0.05), heading=(0., 0)
        ),
        limit_ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            #lin_vel_x=(0, 4.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-2., 2.), heading=(0., 0)
            lin_vel_x=(0, 2.8), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-2., 2.), heading=(0., 0)
        ),
        small2zero_threshold_line=0.15,
        small2zero_threshold_angle=0.05
    )

    def __post_init__(self):
        self.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

from . import obs


@configclass
class ObservationsCfg(obs.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(obs.ObservationsCfg.PolicyCfg):
        symmetry_flags = ObservationTermCfg(func=command_extends.symmetry_flags, params={"command_name": "base_velocity"})

    @configclass
    class CriticCfg(obs.ObservationsCfg.CriticCfg):
        symmetry_flags = ObservationTermCfg(func=command_extends.symmetry_flags, params={"command_name": "base_velocity"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


############################################################################
import torch
import numpy as np

class HistorySymmetry(object):

    '''
    0 left_hip_pitch_joint -0.10000000149011612 -2.5306997299194336 2.8797998428344727
    1 right_hip_pitch_joint -0.10000000149011612 -2.5306997299194336 2.8797998428344727
    2 waist_yaw_joint 0.0 -2.618000030517578 2.618000030517578
    3 left_hip_roll_joint 0.0 -0.5235999226570129 2.967099666595459
    4 right_hip_roll_joint 0.0 -2.967099666595459 0.5235999226570129
    5 left_hip_yaw_joint 0.0 -2.7576000690460205 2.7576000690460205
    6 right_hip_yaw_joint 0.0 -2.7576000690460205 2.7576000690460205
    7 left_knee_joint 0.30000001192092896 -0.08726699650287628 2.8797998428344727
    8 right_knee_joint 0.30000001192092896 -0.08726699650287628 2.8797998428344727
    9 left_shoulder_pitch_joint 0.5 -3.0891997814178467 2.6703999042510986
    10 right_shoulder_pitch_joint 0.5 -3.0891997814178467 2.6703999042510986
    11 left_ankle_pitch_joint -0.20000000298023224 -0.8726699352264404 0.5235999226570129
    12 right_ankle_pitch_joint -0.20000000298023224 -0.8726699352264404 0.5235999226570129
    13 left_shoulder_roll_joint 0.0 -1.5881999731063843 2.251499652862549
    14 right_shoulder_roll_joint 0.0 -2.251499652862549 1.5881999731063843
    15 left_ankle_roll_joint 0.0 -0.26179996132850647 0.26179996132850647
    16 right_ankle_roll_joint 0.0 -0.26179996132850647 0.26179996132850647
    17 left_shoulder_yaw_joint 0.0 -2.618000030517578 2.618000030517578
    18 right_shoulder_yaw_joint 0.0 -2.618000030517578 2.618000030517578
    19 left_elbow_joint 0.30000001192092896 -1.0471998453140259 2.0943996906280518
    20 right_elbow_joint 0.30000001192092896 -1.0471998453140259 2.0943996906280518
    '''

    def __init__(self):
        mirrors_ids = np.array([1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12,11, 14,13, 16,15, 18,17, 20,19], dtype = np.int32)

        policy_ids = [
            np.array([i for i in range(3)], dtype = np.int32), # ang_vel
            np.array([i for i in range(3)], dtype = np.int32) + 3, # gravity
            np.array([i for i in range(3)], dtype = np.int32) + 6, # commands
            mirrors_ids + 9, # joint_pos
            mirrors_ids + 30, # joint_vel
            mirrors_ids + 51, # actions
        ]
        policy_ids = np.concatenate(policy_ids, axis = 0)

        policy_mirrors = np.ones_like(policy_ids, dtype= np.float32)
        policy_mirrors[[1, 4, 7]] = -1

        critic_ids = [
            np.array([i for i in range(3)], dtype = np.int32), # lin_vel
            np.array([i for i in range(3)], dtype = np.int32) + 3, # ang_vel
            np.array([i for i in range(3)], dtype = np.int32) + 6, # gravity
            np.array([i for i in range(3)], dtype = np.int32) + 9, # commands
            mirrors_ids + 12, # joint_pos
            mirrors_ids + 33, # joint_vel
            mirrors_ids + 54, # actions
            mirrors_ids + 75, # joint_acc
            mirrors_ids + 96, # joint_stiffness
            mirrors_ids + 117, # joint_damping
            mirrors_ids + 138, # friction_coeff
            mirrors_ids + 159, # torques
            np.array([1, 0], dtype = np.int32) + 180, # feet_status
            np.array([3, 4, 5,  0, 1, 2], dtype = np.int32) + 182, # feet_forces
            np.array([3, 4, 5,  0, 1, 2], dtype = np.int32) + 188, # feet_pos
        ]
        critic_ids = np.concatenate(critic_ids, axis = 0)

        critic_mirrors = np.ones_like(critic_ids, dtype= np.float32)
        critic_mirrors[[1, 4, 7, 10, 183, 186, 189, 192]] = -1

        self.mirrors_ids = torch.tensor(mirrors_ids)
        self.policy_ids = torch.tensor(policy_ids)
        self.critic_ids = torch.tensor(critic_ids)
        self.policy_mirrors = torch.tensor(policy_mirrors)
        self.critic_mirrors = torch.tensor(critic_mirrors)


    def _setup_history(self, history_length, device):

        # mirrors_list = []
        policy_ids = []
        critic_ids = []

        policy_mirrors = []
        critic_mirrors = []
        for i in range(history_length):
            policy_ids.append(self.policy_ids + i * len(self.policy_ids))
            critic_ids.append(self.critic_ids + i * len(self.critic_ids))

            policy_mirrors.append(self.policy_mirrors)
            critic_mirrors.append(self.critic_mirrors)

        policy_ids = torch.cat(policy_ids)
        critic_ids = torch.cat(critic_ids)
        policy_mirrors = torch.cat(policy_mirrors)
        critic_mirrors = torch.cat(critic_mirrors)

        self.mirrors_ids = self.mirrors_ids.to(device)
        self.policy_ids = policy_ids.to(device)
        self.critic_ids = critic_ids.to(device)
        self.policy_mirrors = policy_mirrors.to(device)
        self.critic_mirrors = critic_mirrors.to(device)

    def mirror_policy(self, obs: torch.Tensor):
        if obs.device != self.policy_ids.device:
            history_length = obs.shape[-1] // self.policy_ids.shape[-1]
            self._setup_history(history_length, obs.device)

        mirror_obs = obs[:, self.policy_ids] * self.policy_mirrors[None, :]
        return mirror_obs

    def mirror_critic(self, obs: torch.Tensor):
        if obs.device != self.critic_ids.device:
            history_length = obs.shape[-1] // self.critic_ids.shape[-1]
            self._setup_history(history_length, obs.device)

        mirror_obs = obs[:, self.critic_ids] * self.critic_mirrors[None, :]
        return mirror_obs

    def mirror_action(self, actions: torch.Tensor):
        ## TODO
        mirror_actions = actions[:, self.mirrors_ids]
        return mirror_actions
