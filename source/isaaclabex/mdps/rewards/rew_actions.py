from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from .bodies import actions

"""
Action Penalties:
Penalties to enforce smoothness and regularity in the agent's actions.
"""
p_action_rate_l2 = isaaclab_rewards.action_rate_l2                  # L2 penalty on the rate of change of actions.
p_action_rate2_l2 = actions.penalize_action_rate2_l2
p_action_rate_norm = spot_rewards.action_smoothness_penalty         # Norm penalty to smooth action changes.
p_action_smoothness = actions.penalize_action_smoothness            # Direct smoothness penalty for actions.
p_action_l2 = isaaclab_rewards.action_l2                            # L2 penalty on action magnitude.
