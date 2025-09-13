# Module for calculating episode rewards based on joint status.
# This module computes rewards based on differences in joint positions and their statistical properties.
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


def _exp_zero(error, error_std, penalize_weight: float = None):
    if penalize_weight is None:
        return torch.exp(- error / error_std)

    return torch.exp(- error / error_std) + penalize_weight * (error / error_std)

from enum import Enum
class mirror_or_synchronize(Enum):
    NONE = 0
    MIRROR = 1
    SYNCHRONIZE = 2

def rew_mean_zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    error_std: float = 0.1,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids]

    reward = _exp_zero(torch.abs(episode_mean), error_std)

    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_mean_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    error_std: float = 0.1
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    # mean diff
    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids]
    means0 = episode_mean[:, ::2]
    means1 = episode_mean[:, 1::2]
    diff = torch.abs(means0 - means1)
    reward = _exp_zero(diff, error_std)  # double weight
    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_mean_step_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.08,
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    pos = term.diff[:, asset_cfg.joint_ids]

    if type == mirror_or_synchronize.MIRROR:
        step_ids = term.step_ids(asset_cfg)
        step_mean_mean = term.step_mean_mean_buf[:, step_ids]
        zeros = (torch.abs(pos[:, ::2] + pos[:, 1::2]) + torch.abs(step_mean_mean)) / 2
        reward = _exp_zero(zeros, error_std)

    elif type == mirror_or_synchronize.SYNCHRONIZE:
        zeros = torch.abs(pos[:, ::2] - pos[:, 1::2]) ###
        reward = _exp_zero(zeros, error_std)

    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

def rew_mean_constraint(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_statistics_name: str = "pos",
    offset_constraint: float = 0.35,
    error_std: float = 0.08
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: joints.StatusJPos = manager.get_term(pos_statistics_name)

    # mean diff
    episode_mean = term.episode_mean_buf[:, asset_cfg.joint_ids]
    means0 = episode_mean[:, ::2]
    means1 = episode_mean[:, 1::2]

    means = [means0, means1]
    constraint_reward = None
    for mean in means:
        mean = torch.abs(mean)
        offset = torch.clamp(mean - offset_constraint, 0, 50)
        offset_reward = _exp_zero(offset, error_std)
        constraint_reward = offset_reward if constraint_reward is None else constraint_reward + offset_reward

    constraint_reward /= len(means)
    reward = constraint_reward
    reward = torch.sum(reward, dim=-1)

    flag = torch.logical_or(term.stand_flag, term.zero_flag)
    diff_reward = torch.exp(-torch.norm(term.diff[:, asset_cfg.joint_ids], dim = -1))
    reward[flag] = diff_reward[flag]
    return reward

