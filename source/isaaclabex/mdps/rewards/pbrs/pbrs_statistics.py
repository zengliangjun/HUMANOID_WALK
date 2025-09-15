from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from .. statistics import mean_joints, var_joints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from . import pbrs_base

class mean_zero(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = mean_joints.rew_mean_zero \
                (env=env, asset_cfg=asset_cfg, pos_statistics_name = pos_statistics_name, error_std = error_std)
        return self._calculate(_penalize)


class mean_symmetry(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",
        type: mean_joints.mirror_or_synchronize = mean_joints.mirror_or_synchronize.NONE,
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = mean_joints.rew_mean_symmetry \
                (env=env, asset_cfg=asset_cfg, pos_statistics_name = pos_statistics_name,
                type=type, error_std = error_std)
        return self._calculate(_penalize)


class mean_step_symmetry(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",

        type: mean_joints.mirror_or_synchronize = mean_joints.mirror_or_synchronize.NONE,
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = mean_joints.rew_mean_step_symmetry \
                (env=env, asset_cfg=asset_cfg, pos_statistics_name = pos_statistics_name,
                 type=type, error_std = error_std)

        return self._calculate(_penalize)


class mean_constraint(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",

        offset_constraint: float = 0.35,
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = mean_joints.rew_mean_constraint \
                (env=env, asset_cfg=asset_cfg,
                 pos_statistics_name = pos_statistics_name,
                 offset_constraint = offset_constraint,
                 error_std = error_std)

        return self._calculate(_penalize)

## variance

class variance_zero(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = var_joints.rew_variance_zero \
                (env=env, asset_cfg=asset_cfg,
                 pos_statistics_name = pos_statistics_name,
                 error_std = error_std)

        return self._calculate(_penalize)

class variance_symmetry(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",
        type: mean_joints.mirror_or_synchronize = mean_joints.mirror_or_synchronize.NONE,
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = var_joints.rew_variance_symmetry \
                (env=env, asset_cfg=asset_cfg,
                 pos_statistics_name = pos_statistics_name,
                 type=type,
                 error_std = error_std)

        return self._calculate(_penalize)

class variance_constraint(pbrs_base.PbrsBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,

        asset_cfg: SceneEntityCfg,
        pos_statistics_name: str = "pos",
        max_constraint: float = 0.09,
        min_constraint: float = 0.01,
        error_std: float = 0.1,

        method: int = pbrs_base.PBRSNormal,
        sigma: float = 0.25,
        gamma: float = 1,
        ) -> torch.Tensor:

        _penalize = var_joints.rew_variance_constraint \
                (env=env, asset_cfg=asset_cfg,
                 pos_statistics_name = pos_statistics_name,
                 max_constraint = max_constraint,
                 min_constraint = min_constraint,
                 error_std = error_std)

        return self._calculate(_penalize)
