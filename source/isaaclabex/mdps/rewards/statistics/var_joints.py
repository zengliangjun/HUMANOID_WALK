from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager
    from isaaclabex.mdps.statistics import joints

from .mean_joints import _exp_zero, mirror_or_synchronize

def rew_variance_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    error_std: float = 0.004,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]

    reward = _exp_zero(episode_variance, error_std)
    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_variance_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",

    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.1,

) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]
    episode_variance0 = episode_variance[:, ::2]
    episode_variance1 = episode_variance[:, 1::2]

    ##
    diff = torch.abs(episode_variance0 - episode_variance1)
    reward = _exp_zero(diff, error_std)

    if type == mirror_or_synchronize.MIRROR:
        step_ids = term.step_ids(asset_cfg)

        ## mean variance is zero
        step_ids = term.step_ids(asset_cfg)
        step_mean_variance = term.step_mean_variance_buf[:, step_ids]
        reward += _exp_zero(step_mean_variance, error_std)

    elif type == mirror_or_synchronize.SYNCHRONIZE:
        step_ids = term.step_ids(asset_cfg)

        ## mean variance equal to variance
        step_ids = term.step_ids(asset_cfg)
        step_mean_variance = term.step_mean_variance_buf[:, step_ids]
        step_variance_mean = term.step_variance_mean_buf[:, step_ids]

        diff = (torch.square(episode_variance0 - step_mean_variance) + torch.square(episode_variance1 - step_mean_variance)) / 2
        diff = (torch.sqrt(diff) + step_variance_mean) / 2
        reward += _exp_zero(diff, error_std)

    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_variance_constraint(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",

    max_constraint: float = 0.09,
    min_constraint: float = 0.01,
    error_std: float = 0.1,

) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_variance = term.episode_variance_buf[:, asset_cfg.joint_ids]

    diffmin = torch.clamp(episode_variance - min_constraint, -50, 0)
    diffmax = torch.clamp(episode_variance - max_constraint, 0, 50)
    ##
    error = torch.abs(diffmin) + diffmax
    reward = _exp_zero(error, error_std)
    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward


