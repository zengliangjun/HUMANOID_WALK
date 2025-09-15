from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.cartpole.mdp import rewards as cartpole_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from .bodies import base, bodies

"""
Base Rewards and Penalties:
Reward and penalty values pertaining to the base's orientation, height, and velocity tracking.
"""
# Orientation rewards/penalties: penalize orientation deviation measured by different metrics.
p_ori_l2 = isaaclab_rewards.flat_orientation_l2                # L2 penalty for flat orientation error.
p_ori_norm = spot_rewards.base_orientation_penalty             # Norm penalty for orientation error.
rew_ori_gravity = humanoid_rewards.upright_posture_bonus            # Bonus for maintaining upright gravity alignment.
rew_ori_euler_gravity_b = base.reward_orientation_euler_gravity_b  # Euler-based orientation reward.

# Height-based rewards/penalties.
p_height_flat_or_rayl2 = isaaclab_rewards.base_height_l2         # L2 penalty on base height error (flat/ray).
p_height_base2feet = base.penalize_base_height                   # Penalty for mismatch in base-to-feet height.
rp_base_height = base.reward_penalize_base_height
rp_height_upper = base.reward_penalize_height_upper

"""
Body Rewards:
Penalties and rewards specific to the body dynamics.
"""
p_body_lin_acc_l2 = isaaclab_rewards.body_lin_acc_l2               # L2 penalty on body linear acceleration.
rew_body_distance = bodies.reward_distance                              # Reward based on body distance metric.
rew_body_distance_b = bodies.reward_distance_b                              # Reward based on body distance metric.
rew_width = bodies.reward_width                                 # Reward based on body width metric.
p_width = bodies.penalize_width

rew_stability = bodies.Stability


p_undesired_contacts = isaaclab_rewards.undesired_contacts           # Penalty for undesired contacts.
penalize_contacts = bodies.penalize_collision

rew_pitch_total2zero = bodies.rew_pitch_total2zero



from .pbrs import pbrs_bodies

pbrs_width = pbrs_bodies.width
