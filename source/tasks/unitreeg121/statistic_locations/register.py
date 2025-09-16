import gymnasium as gym

from . import ppo_cfg
from . import ppo_cfg, env_cfg

gym.register(
    id="G121ObsStatistic-rnn",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfgRNN",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgRNN",
    },
)

gym.register(
    id="G121ObsStatistic-Play-rnn",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfgRNN_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgRNN",
    },
)

gym.register(
    id="G121ObsStatistic-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfgHistory",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgHistory",
    },
)

gym.register(
    id="G121ObsStatistic-Play-history",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{env_cfg.__name__}:G1ObsStatisticsCfgHistory_PLAY",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ObsStatisticCfgHistory",
    },
)
