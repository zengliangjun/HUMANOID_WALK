from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards
from .feet import contact, feet

'''
Feet Rewards and Penalties:
These parameters handle rewards and penalties related to feet dynamics, such as air time, sliding, and clearance.
'''
# Air time rewards/penalties.
rew_air_time = loc_rewards.feet_air_time                               # Reward for being airborne.

rew_air_time_biped = loc_rewards.feet_air_time_positive_biped           # Reward for biped air time.
rew_air_time2 = spot_rewards.air_time_reward                            # Alternative air time reward.

p_airborne = contact.penalty_feet_airborne                      # Penalty for feet being airborne too long.
p_both_feet_in_air = contact.penalize_both_feet_in_air
p_air_time_variance = spot_rewards.air_time_variance_penalty           # Penalty based on variability in air time.

# Slide penalties.
p_slide = loc_rewards.feet_slide                                    # Penalty for feet sliding.
p_slide_threshold = contact.penalize_feet_slide                         # Penalty applied at sliding threshold.
p_slide_threshold2 = spot_rewards.foot_slip_penalty                   # Additional slip penalty.
p_stumble = contact.penalty_feet_stumble                         # Penalty for stumbling.

# feet height
p_max_feet_height_before_contact = contact.penalize_max_feet_height_before_contact
p_clearance = contact.penalize_foot_clearance                           # Penalty for insufficient clearance.
rew_clearance = spot_rewards.foot_clearance_reward                      # Reward for adequate foot clearance.

# Gait reward.
GaitReward = spot_rewards.GaitReward                                       # Reward for achieving desired gait pattern.

p_feet_orientation = feet.penalize_feet_orientation

p_forces = isaaclab_rewards.contact_forces                           # Penalty for excessive contact forces.
p_forces_z = contact.penalize_feet_forces_z                        # Reward based on vertical forces.
p_forces2 = contact.penalize_feet_forces                        # Additional penalty for contact forces.


