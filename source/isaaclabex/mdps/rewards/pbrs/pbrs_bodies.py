from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from .. bodies import bodies

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from . import pbrs_base


class width(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        target_width: float = 0.2,
        target_height: float = 0.78,
        center_velocity: float = 0.4,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = - bodies.penalize_width \
                (env=env, asset_cfg=asset_cfg,
                target_width = target_width, target_height = target_height, center_velocity = center_velocity)
        return self._calculate(_penalize)
