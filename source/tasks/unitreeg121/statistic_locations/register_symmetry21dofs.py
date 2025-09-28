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

gym.register(
    id="G121SymmetryRNN_PLAY",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgRNN",
    },
)

gym.register(
    id="G121SymmetryRNNV2",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1CfgV2",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgV2RNN",
    },
)

gym.register(
    id="G121SymmetryRNNV2_PLAY",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1CfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgV2RNN",
    },
)


gym.register(
    id="G121SymmetryRNNV3",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1CfgV3",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgV3RNN",
    },
)

gym.register(
    id="G121SymmetryRNNV3_PLAY",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg_symmetry21dofs.__name__}:G1CfgV3_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg_symmetry21dofs.__name__}:G1CfgV3RNN",
    },
)