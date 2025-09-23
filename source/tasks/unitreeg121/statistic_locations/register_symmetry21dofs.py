import gymnasium as gym

from . import ppo_cfg_symmetry21dofs, env_cfg_symmetry21dofs

gym.register(
    id="G121SymmetryRNN",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1Cfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgRNN",
    },
)
