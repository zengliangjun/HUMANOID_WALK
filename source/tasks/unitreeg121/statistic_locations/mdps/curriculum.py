from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclabex.mdps.curriculum import events, adaptive
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)

    events_with_steps = CurriculumTermCfg(
        func=events.range_with_degree,
        params={
            "degree": 0.0000001,
            "down_up_lengths":[800, 900],
            "scale_range": [0, 1],
            "scale": 0,
            "manager_name": "event",
            "curriculums": {
                'startup_material': {    # event name
                    "static_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.05, 1.5)
                    ),
                    "dynamic_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.8, 1.2)
                    )
                },
                'reset_base': {    # event name
                    "pose_range": events.EventCurriculumStepItem(
                        start_range = {"x": (-0, 0), "y": (-0, 0), "yaw": (-0, 0)},
                        end_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
                    ),
                    "velocity_range": events.EventCurriculumStepItem(
                        start_range =  {
                                            "x": (-0.0, 0.),
                                            "y": (-0.0, 0.0),
                                            "z": (-0.0, 0.0),
                                            "roll": (-0., 0.0),
                                            "pitch": (-0.0, 0.0),
                                            "yaw": (-0.0, 0.0),
                                        },
                        end_range =  {
                                        "x": (-0.2, 0.2),
                                        "y": (-0.2, 0.2),
                                        "z": (-0.2, 0.2),
                                        "roll": (-0.2, 0.2),
                                        "pitch": (-0.2, 0.2),
                                        "yaw": (-0.2, 0.2),
                                    }
                    ),
                },
                'reset_joints': {    # event name
                    "position_range": events.EventCurriculumStepItem(
                        start_range = (0, 0),
                        end_range = (0.5, 1.5)
                    ),
                },
                'interval_push': {    # event name
                    "velocity_range": events.EventCurriculumStepItem(
                        start_range =  {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
                        end_range =  {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
                    ),
                },
                'interval_gravity': {    # event name
                    "gravity_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (0, 0),
                        end_range =  (-0.1, 0.1)
                    ),
                },
                'interval_actuator': {    # event name
                    "stiffness_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (.8, 1.2)
                    ),
                    "damping_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (.8, 1.2)
                    ),
                },
                'interval_mass': {    # event name
                    "mass_distribution_params": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.85, 1.15)
                    ),
                },
                'interval_coms': {    # event name
                    "coms_distribution_params": events.EventCurriculumStepItem(
                        start_range = (0, 0),
                        end_range = (-0.15, 0.15)
                    ),
                },
            }
        },
    )


    penalize_steps = CurriculumTermCfg(
        func=adaptive.scale_with_degree,
        params={
            'degree': 0.0000001,
            'down_up_lengths': [450, 700],
            "scale_range": [0, 1],
            "scale": 0,
            "manager_name": "reward",
            "curriculums": {
                'p_linear_velocity': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.5,
                    "end_weight": -2.0
                },
                'p_angular_velocity': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.008,
                    "end_weight": -0.05
                },
                'p_action_rate_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -5e-3,
                    "end_weight": -2e-2
                },
                'p_action_rate_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -5e-3,
                    "end_weight": -2e-2
                },
                'p_action_rate_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -5e-3,
                    "end_weight": -2e-2
                },
                'p_action_rate_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -5e-3,
                    "end_weight": -2e-2
                },
                'p_action_smoothness_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -8e-4,
                    "end_weight": -4e-3
                },
                'p_action_smoothness_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -8e-4,
                    "end_weight": -4e-3
                },
                'p_action_smoothness_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -8e-4,
                    "end_weight": -4e-3
                },
                'p_action_smoothness_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -8e-4,
                    "end_weight": -4e-3
                },
                'p_energy_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -4e-6,
                    "end_weight": -2e-5
                },
                'p_energy_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -4e-6,
                    "end_weight": -2e-5
                },
                'p_energy_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -4e-6,
                    "end_weight": -2e-5
                },
                'p_energy_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -4e-6,
                    "end_weight": -2e-5
                },
                'p_pos_limits_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1.0,
                    "end_weight": -5.0
                },
                'p_pos_limits_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1.0,
                    "end_weight": -5.0
                },
                'p_pos_limits_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1.0,
                    "end_weight": -5.0
                },
                'p_pos_limits_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1.0,
                    "end_weight": -5.0
                },
                'p_jvel_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2e-4,
                    "end_weight": -1e-3
                },
                'p_jvel_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2e-4,
                    "end_weight": -1e-3
                },
                'p_jvel_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2e-4,
                    "end_weight": -1e-3
                },
                'p_jvel_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2e-4,
                    "end_weight": -1e-3
                },
                'p_jacc_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2.5e-8,
                    "end_weight": -2.5e-7
                },
                'p_jacc_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2.5e-8,
                    "end_weight": -2.5e-7
                },
                'p_jacc_LS': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2.5e-8,
                    "end_weight": -2.5e-7
                },
                'p_jacc_LD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2.5e-8,
                    "end_weight": -2.5e-7
                },
                'p_deviation_UD': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.16,
                    "end_weight": -0.8
                },
                'p_deviation_US': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.2,
                    "end_weight": -1
                },
                'p_deviation_legs': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.2, #-0.01,
                    "end_weight": -1
                },
                'p_legwidth': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1,
                    "end_weight": -5
                },
                'p_orientation': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1,
                    "end_weight": -5
                },
                'p_feet_vertical_force': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1e-3,
                    "end_weight": -5e-3
                },
                'p_feet_vertical_force': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1e-3,
                    "end_weight": -5e-3
                },
                'p_feet_slide': {    # reward name
                    "param_name": "weight",
                    "start_weight": -3e-2,
                    "end_weight": -0.1
                },
                'p_feet_clearance': {    # reward name
                    "param_name": "weight",
                    "start_weight": -3e-2,
                    "end_weight": -1.0
                },
                'p_feet_ori': {    # reward name
                    "param_name": "weight",
                    "start_weight": -2e-2,
                    "end_weight": -0.1
                },
                'rew_alive': {    # reward name
                    "param_name": "weight",
                    "start_weight": 3e-2,
                    "end_weight": 0.1
                },
            }
        }
    )


@configclass
class CurriculumCfg2(CurriculumCfg):
    reward_leg_steps = CurriculumTermCfg(
        func=adaptive.scale_with_degree,
        params={
            'degree': 0.00003,
            'down_up_lengths': [120, 150],
            "scale_range": [0, 1],
            "scale": 0,
            "manager_name": "reward",
            "curriculums": {
                'rew_var_leg_constraint': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0.0015,
                    "end_weight": 0.02
                }
            }
        }
    )
