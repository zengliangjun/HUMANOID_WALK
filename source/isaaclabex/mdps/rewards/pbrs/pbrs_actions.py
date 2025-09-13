from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from .. bodies import actions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from . import pbrs_base

class action_smoothness(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.term = actions.penalize_action_smoothness(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        weight1: float = 1,    # Weight for penalizing first-order difference (current vs previous)
        weight2: float = 1,    # Weight for penalizing second-order difference (acceleration)
        weight3: float = 0.05, # Weight for penalizing absolute action magnitude

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = - self.term(env=env, asset_cfg=asset_cfg, weight1 = weight1, weight2 = weight2, weight3 = weight3)
        return self._calculate(_penalize)
