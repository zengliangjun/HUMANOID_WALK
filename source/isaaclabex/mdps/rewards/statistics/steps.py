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
    from isaaclabex.mdps.statistics import steps

from .mean_joints import _exp_zero, mirror_or_synchronize

def rew_feetpose_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    statistics_name: str = "steps",
    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.1
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: steps.StatusStep = manager.get_term(statistics_name)

    variance_buf = torch.sqrt(term.feetpose3d_variance_buf)
    mean_buf = term.feetpose3d_mean_buf

    meanx = torch.abs(mean_buf[..., ::2, 0] - mean_buf[..., 1::2, 0])
    meany = torch.abs(mean_buf[..., ::2, 1] + mean_buf[..., 1::2, 1])

    rewmean = (_exp_zero(meanx, error_std) + _exp_zero(meany, error_std)) / 2

    variancex = torch.abs(variance_buf[..., ::2, 0] - variance_buf[..., 1::2, 0])
    variancey = torch.abs(variance_buf[..., ::2, 1] + variance_buf[..., 1::2, 1])

    rewvar = (_exp_zero(variancex, error_std) + _exp_zero(variancey, error_std)) / 2
    rew = torch.sum((rewmean + rewvar) / 2, dim = -1)

    steps_falg = torch.sum(term.air_steps_buf, dim = -1) < 2
    flag = torch.logical_or(term.stand_flag, steps_falg)
    rew[flag] = 0
    return rew

def rew_bodiespose_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    statistics_name: str = "steps",
    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.1
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: steps.StatusStep = manager.get_term(statistics_name)

    ids = term.statistics_bodiesids(asset_cfg)

    variance_buf = torch.sqrt(term.bodiespose3d_variance_buf[:, ids])
    mean_buf = term.bodiespose3d_mean_buf[:, ids]

    meanx = torch.abs(mean_buf[..., ::2, 0] - mean_buf[..., 1::2, 0])
    meany = torch.abs(mean_buf[..., ::2, 1] + mean_buf[..., 1::2, 1])

    rewmean = (_exp_zero(meanx, error_std) + _exp_zero(meany, error_std)) / 2

    variancex = torch.abs(variance_buf[..., ::2, 0] - variance_buf[..., 1::2, 0])
    variancey = torch.abs(variance_buf[..., ::2, 1] + variance_buf[..., 1::2, 1])

    rewvar = (_exp_zero(variancex, error_std) + _exp_zero(variancey, error_std)) / 2

    rew = torch.sum((rewmean + rewvar) / 2, dim = -1)

    steps_falg = torch.sum(term.air_steps_buf, dim = -1) < 2
    flag = torch.logical_or(term.stand_flag, steps_falg)
    rew[flag] = 0
    return rew

def rew_jointspose_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    statistics_name: str = "steps",
    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.1
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: steps.StatusStep = manager.get_term(statistics_name)

    ids = term.statistics_jointsids(asset_cfg)

    variance_buf = torch.sqrt(term.jointspose_variance_buf[:, ids])
    mean_buf = term.jointspose_mean_buf[:, ids]

    mean = torch.abs(mean_buf[..., ::2] - mean_buf[..., 1::2])
    rewmean = _exp_zero(mean, error_std)

    variance = torch.abs(variance_buf[..., ::2] - variance_buf[..., 1::2])
    rewvar = _exp_zero(variance, error_std)

    rew = torch.sum((rewmean + rewvar) / 2, dim = -1)

    steps_falg = torch.sum(term.air_steps_buf, dim = -1) < 2
    flag = torch.logical_or(term.stand_flag, steps_falg)
    rew[flag] = 0
    return rew

def rew_times_symmetry(
    env: ManagerBasedRLEnv,
    statistics_name: str = "steps",
    type: mirror_or_synchronize = mirror_or_synchronize.NONE,
    error_std: float = 0.1
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: steps.StatusStep = manager.get_term(statistics_name)

    air_variance_buf = torch.sqrt(term.air_variance_buf)
    air_mean_buf = term.air_mean_buf
    contact_variance_buf = torch.sqrt(term.contact_variance_buf)
    contact_mean_buf = term.contact_mean_buf

    mean = torch.abs(contact_mean_buf[..., ::2] - contact_mean_buf[..., 1::2]) + \
           torch.abs(air_mean_buf[..., ::2] - air_mean_buf[..., 1::2])
    rewmean = _exp_zero(mean / 2, error_std)

    variance = (air_variance_buf[..., ::2] + air_variance_buf[..., 1::2]) / 2 + \
               (contact_variance_buf[..., ::2] + contact_variance_buf[..., 1::2])
    rewvar = _exp_zero(variance / 2, error_std)

    rew = torch.sum((rewmean + rewvar) / 2, dim = -1)

    steps_falg = torch.sum(term.air_steps_buf, dim = -1) < 2
    flag = torch.logical_or(term.stand_flag, steps_falg)
    rew[flag] = 0
    return rew

