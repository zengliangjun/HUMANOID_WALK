from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def energy_cost(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    '''
    penalizes output torques to reduce energy consumption.
    '''

    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = asset.data.applied_torque
    joint_vel = asset.data.joint_vel

    return torch.sum(torch.abs(joint_torque * joint_vel), dim = -1)
