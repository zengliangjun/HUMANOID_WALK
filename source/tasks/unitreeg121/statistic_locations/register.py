import gymnasium as gym

from . import ppo_cfg
from . import ppo_cfg, env_cfg

gym.register(
    id="G1PBRSCfgRNN-rnn",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1PBRSCfgRNN",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1PBRSCfgRNN",
    },
)

gym.register(
    id="G1PBRSCfgRNN-Play-rnn",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1PBRSCfgRNN_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1PBRSCfgRNN",
    },
)

gym.register(
    id="G1PBRSCfgHistory-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1PBRSCfgHistory",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1PBRSCfgHistory",
    },
)

gym.register(
    id="G1PBRSCfgHistory-Play-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1PBRSCfgHistory_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1PBRSCfgHistory",
    },
)

gym.register(
    id="G1NormalCfgHistory-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1NormalCfgHistory",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1NormalCfgHistory",
    },
)

gym.register(
    id="G1NormalCfgHistory-Play-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1NormalCfgHistory_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1NormalCfgHistory",
    },
)
