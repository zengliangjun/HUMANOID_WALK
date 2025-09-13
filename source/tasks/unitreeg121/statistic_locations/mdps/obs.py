from isaaclab.utils import configclass
from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg, \
                              ObservationGroupCfg, SceneEntityCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.mdps.observations import statistics, privileged

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        ## proprioceptive
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale = 0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        commands = ObservationTermCfg(func=mdp.generated_commands, scale = 0.25, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale = 0.05, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale = 2.0)
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale = 0.25)
        gravity = ObservationTermCfg(func=mdp.projected_gravity)
        commands = ObservationTermCfg(func=mdp.generated_commands, scale = 0.25, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale = 0.05)
        actions = ObservationTermCfg(func=mdp.last_action)

        joint_acc = ObservationTermCfg(func=privileged.joint_acc,
                                       params={"asset_cfg": SceneEntityCfg("robot")},
                                       scale = 0.05)
        joint_stiffness = ObservationTermCfg(func=privileged.joint_stiffness,
                                       params={"asset_cfg": SceneEntityCfg("robot")},
                                       scale = 0.05)
        joint_damping = ObservationTermCfg(func=privileged.joint_damping,
                                       params={"asset_cfg": SceneEntityCfg("robot")},
                                       scale = 0.5)
        friction_coeff = ObservationTermCfg(func=privileged.joint_friction_coeff,
                                       params={"asset_cfg": SceneEntityCfg("robot")},
                                       )
        torques = ObservationTermCfg(func=privileged.joint_torques,
                                       params={"asset_cfg": SceneEntityCfg("robot")},
                                       scale = 1e-5)

        feet_status = ObservationTermCfg(func=privileged.feet_contact_status,
                                        params={
                                                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                                                "threshold": 1.0,
                                            },
                                       )
        feet_forces = ObservationTermCfg(func=privileged.feet_contact_forces,
                                       params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
                                       scale = 1e-5)
        feet_pos = ObservationTermCfg(func=privileged.feet_pos,
                                       params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
                                       )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class ActionStatisticsCfg(ObservationGroupCfg):
        action_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        action_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "action"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PosStatisticsCfg(ObservationGroupCfg):
        pos_episode_mean = ObservationTermCfg(func=statistics.obs_episode_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_episode_variance = ObservationTermCfg(func=statistics.obs_episode_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_mean_mean = ObservationTermCfg(func=statistics.obs_step_mean_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_mean_variance = ObservationTermCfg(func=statistics.obs_step_mean_variance,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_step_variance_mean = ObservationTermCfg(func=statistics.obs_step_variance_mean,
                                       params={"pos_statistics_name": "pos"},
                                       noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    def __post_init__(self):
        self.policy.phase = None
        self.critic.phase = None

    # observation groups
    action_statistics: ActionStatisticsCfg = ActionStatisticsCfg()
    pos_statistics: PosStatisticsCfg = PosStatisticsCfg()

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
