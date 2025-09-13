from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from .bodies import task

# Velocity tracking rewards/penalties.
p_lin_z_l2 = isaaclab_rewards.lin_vel_z_l2                      # L2 penalty for linear velocity (z).
p_ang_xy_l2 = isaaclab_rewards.ang_vel_xy_l2                    # L2 penalty for angular velocity (xy).
p_motion_lin_ang = spot_rewards.base_motion_penalty              # Penalty for error in combined linear and angular velocities.
rew_motion_lin_ang = task.reward_mismatch_vel_exp                    # Exponential penalty for velocity mismatch.
rew_motion_speed = task.reward_mismatch_speed                       # Reward based on matching speed.
rew_motion_hard = task.reward_track_vel_hard                        # Hard matching reward for velocity tracking.

# Additional velocity tracking rewards.
rew_lin_xy_exp = isaaclab_rewards.track_lin_vel_xy_exp                # Exponential reward for linear xy velocity tracking.
rew_ang_z_exp = isaaclab_rewards.track_ang_vel_z_exp                  # Exponential reward for angular z velocity tracking.
rew_lin_xy_exp2 = loc_rewards.track_lin_vel_xy_yaw_frame_exp           # Alternative linear velocity reward in yaw frame.
rew_ang_z_exp2 = loc_rewards.track_ang_vel_z_world_exp                 # Alternative angular velocity reward in world frame.
rew_lin_xy_exp3 = spot_rewards.base_linear_velocity_reward            # Additional linear velocity reward sample.
rew_ang_z_exp3 = spot_rewards.base_angular_velocity_reward             # Additional angular velocity reward sample.

# Target position rewards.
rew_move_to_target_bonus = humanoid_rewards.move_to_target_bonus        # Bonus for reaching target.
rew_progress_reward = humanoid_rewards.progress_reward                  # Reward based on progression.

# Base acceleration regularization.
p_base_acc_exp_norm = task.reward_base_acc                           # Reward to regularize base acceleration.
