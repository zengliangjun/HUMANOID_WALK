from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.cartpole.mdp import rewards as cartpole_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from .joints import energy, deviation

"""
Joint Penalties:
Penalties for energy consumption, torque, joint positions, velocities, and acceleration.
"""
# Energy and torque related penalties.
p_torques_l2 = isaaclab_rewards.joint_torques_l2                   # L2 penalty for joint torques.
p_torques_norm = spot_rewards.joint_torques_penalty                # Norm penalty for joint torques.
p_action_vel = humanoid_rewards.power_consumption                # Penalty on power consumption (action x velocity).
p_energy = energy.energy_cost                                     # Energy cost penalty from joint actions.
p_torque_limits = isaaclab_rewards.applied_torque_limits           # Penalty for exceeding torque limits.

# Joint position constraints.
p_jpos_limits_l1 = isaaclab_rewards.joint_pos_limits               # L1 penalty for joint position limits.
try:
    p_jpos_limits_ratio = humanoid_rewards.joint_pos_limits_penalty_ratio  # Preferred joint limits penalty ratio.
except:
    p_jpos_limits_ratio = humanoid_rewards.joint_limits_penalty_ratio        # Fallback joint limits penalty ratio.
# Joint position regularization.
p_jpos_deviation_l1 = isaaclab_rewards.joint_deviation_l1           # L1 penalty for deviation from desired joint positions.
p_jpos_norm_stand_check = spot_rewards.joint_position_penalty        # Norm penalty for joint positions.
rp_joint_deviation = deviation.reward_penalize_joint_deviation            # Reward for correct yaw/roll in joint positions.


# Target joint positions.
p_jpos_target_l2 = cartpole_rewards.joint_pos_target_l2             # L2 penalty to drive joint positions toward targets.

# Joint velocity penalties.
p_jvel_l1 = isaaclab_rewards.joint_vel_l1                          # L1 penalty for joint velocities.
p_jvel_l2 = isaaclab_rewards.joint_vel_l2                          # L2 penalty for joint velocities.
p_jvel_limits = isaaclab_rewards.joint_vel_limits                   # Penalty for exceeding velocity limits.
p_jvel_norm = spot_rewards.joint_velocity_penalty                   # Norm penalty for joint velocity errors.

# Joint acceleration regularization.
p_jacc_l2 = isaaclab_rewards.joint_acc_l2                           # L2 penalty for joint accelerations.
p_jacc_norm = spot_rewards.joint_acceleration_penalty               # Norm penalty for joint acceleration errors.


