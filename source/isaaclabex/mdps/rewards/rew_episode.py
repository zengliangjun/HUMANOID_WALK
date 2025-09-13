from isaaclab.envs.mdp import rewards as isaaclab_rewards

'''
Episodic Rewards:
These parameters reward or penalize the high-level episode state,
such as whether the agent is still alive or if the simulation has terminated.
'''
rew_eps_alive = isaaclab_rewards.is_alive                       # Reward for being alive.
p_eps_terminated = isaaclab_rewards.is_terminated             # Penalty when terminated abnormally.
p_eps_terminated_term = isaaclab_rewards.is_terminated_term     # Penalty when terminated in terminal state.
