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
    from isaaclabex.mdps.statistics import bodies

from .mean_joints import _exp_zero, mirror_or_synchronize

def rew_bodies_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    bodies_statistics_name: str = "bodies",
    std_ranges: tuple[float] = [0.14, 0.21],
    error_std: float = 0.06
) -> torch.Tensor:

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    term: bodies.StatusPose3d = manager.get_term(bodies_statistics_name)

    sids = term.statistics_ids(asset_cfg)

    episode_mean = term.episode_mean_buf[:, sids]
    episode_variance = term.episode_variance_buf[:, sids]
    episode_std = torch.sqrt(episode_variance)

    mean0 = episode_mean[:, ::2, :]
    mean1 = episode_mean[:, 1::2, :]

    meanx = torch.abs(mean0[..., 0] - mean1[..., 0])
    rewmeanx = _exp_zero(meanx, error_std)
    rewmeanx = torch.sum(rewmeanx, dim = -1)

    '''
    # walk
    stdx = torch.abs(torch.clamp_max(episode_std[..., 0] - std_ranges[0], 0)) + \
                     torch.clamp_min(episode_std[..., 0] - std_ranges[1], 0)
    rew_walkstdx = _exp_zero(stdx, error_std)
    rew_walkstdx = torch.sum(rew_walkstdx, dim = -1) / 2
    # stand
    rew_standstdx = _exp_zero(episode_std[..., 0], error_std)
    rew_standstdx = torch.sum(rew_standstdx, dim = -1) / 2

    stand_flag = torch.logical_or(term.stand_flag, term.zero_flag)
    rewmeanx[stand_flag] += rew_standstdx[stand_flag]

    walkflag = torch.logical_not(stand_flag)
    rewmeanx[walkflag] += rew_walkstdx[walkflag]
    '''

    return rewmeanx
