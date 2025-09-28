from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class JointsBase(ManagerTermBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg
        self.command_name = cfg.params["command_name"]

        self._init_buffers()
        # 初始化标志位

    def _init_buffers(self):
        # 初始化足接触统计的均值与方差缓冲区 (一维数据)
        posecount = len(self.asset_cfg.joint_ids)
        self.episode_variance_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return

        # 清空所有足接触统计缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

    def _update_flag(self):
        command: UniformVelocityCommand = self._env.command_manager.get_term(self.command_name)
        self.stand_flag = torch.logical_or(command.is_standing_env ,
                                           self._env.episode_length_buf <= 1)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        episode_length_buf = self._env.episode_length_buf

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None]

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None] - 1)

        # 当episode刚开始时重置方差，防止数值异常
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _calcute_pose(self, with_default_or_zero):
        self.default_diff = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids] - self.asset.data.default_joint_pos[:, self.asset_cfg.joint_ids]
        if 0 == with_default_or_zero:
            pos = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
        else:
            pos = self.default_diff

        self.calculate_diff = pos
        self._calculate_episode(pos)

    def _calculate_zero(self, error, error_std):
        return torch.exp(- torch.square(error / error_std))

    def _calculate_penalize(self, error, error_std):
        offset = torch.abs(torch.abs(error) - error_std * 0.6)
        penalize = torch.clamp_max(0.5 - offset / (error_std* 0.6), max = 0)
        return torch.square(penalize)

    def _calculate_stand(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):
        # stand
        stand_reward = self._calculate_zero(self.default_diff[self.stand_flag ], error_std) + \
            penalize_weight * self._calculate_penalize(self.default_diff[self.stand_flag ], error_std)
        stand_reward = torch.sum(stand_reward, dim = -1)
        return stand_reward


class JointsSymmetry(JointsBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calculate_mean(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        # mean
        episode_mean = self.episode_mean_buf

        reward_mean = self._calculate_zero(episode_mean, error_std) + \
            penalize_weight * self._calculate_penalize(episode_mean, error_std)

        # current
        diff = self.calculate_diff[:, ::2] + self.calculate_diff[:, 1::2]

        reward_cur = self._calculate_zero(diff, error_std) + \
            penalize_weight * self._calculate_penalize(diff, error_std)
        # sum
        reward = (torch.sum(reward_mean, dim = -1) * 0.6 + torch.sum(reward_cur, dim = -1) * 0.4)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def _calculate_var(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        episode_variance = self.episode_variance_buf
        episode_std = torch.sqrt(episode_variance)

        #
        error = episode_std[:, ::2] - episode_std[:, 1::2]
        reward = self._calculate_zero(error, error_std) + \
            penalize_weight * self._calculate_penalize(error, error_std)
        reward = torch.sum(reward, dim = -1)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def _calculate_var_constraint(self, error_std: float = 0.06,
            penalize_weight: float = -0.25,
            max_constraint: float = 0.09,
            min_constraint: float = 0.01):

        episode_variance = self.episode_variance_buf
        episode_std = torch.sqrt(episode_variance)

        diffmin = torch.clamp(episode_std - min_constraint, -50, 0)
        diffmax = torch.clamp(episode_std - max_constraint, 0, 50)
        ##
        error = torch.abs(diffmin) + diffmax
        reward =  self._calculate_zero(error, error_std) + \
            penalize_weight * self._calculate_penalize(error, error_std)

        reward = torch.sum(reward, dim = -1)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def __call__(self,
            env: ManagerBasedRLEnv,
            command_name: str = "base_velocity",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            with_default_or_zero: int = 0, # 0 is zero, 1 is default
            error_std: float = 0.06,
            penalize_weight: float = -0.25,
            max_constraint: float = 0.09,
            min_constraint: float = 0.01,
            ratios: list[float] = [0.45, 0.45, 0.1],
        ) -> torch.Tensor:
        self._update_flag()
        self._calcute_pose(with_default_or_zero)
        return self._calculate_mean(error_std, penalize_weight) * ratios[0] \
               + self._calculate_var(error_std, penalize_weight) * ratios[1] \
               + self._calculate_var_constraint(error_std, penalize_weight, max_constraint, min_constraint) * ratios[2]


class JointsZero(JointsBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _calculate_mean(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        # mean
        episode_mean = self.episode_mean_buf

        reward_mean = self._calculate_zero(episode_mean, error_std) + \
            penalize_weight * self._calculate_penalize(episode_mean, error_std)

        reward_cur = self._calculate_zero(self.calculate_diff, error_std) + \
            penalize_weight * self._calculate_penalize(self.calculate_diff, error_std)
        # sum
        reward = (torch.sum(reward_mean, dim = -1) * 0.6 + torch.sum(reward_cur, dim = -1) * 0.4)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def _calculate_var(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        episode_variance = self.episode_variance_buf
        episode_std = torch.sqrt(episode_variance)

        reward = self._calculate_zero(episode_std, error_std) + \
            penalize_weight * self._calculate_penalize(episode_std, error_std)
        reward = torch.sum(reward, dim = -1)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def __call__(self,
            env: ManagerBasedRLEnv,
            command_name: str = "base_velocity",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            with_default_or_zero: int = 0, # 0 is zero, 1 is default
            error_std: float = 0.06,
            penalize_weight: float = -0.25,
            ratios: list[float] = [0.5, 0.5, 0.1],
        ) -> torch.Tensor:

        self._update_flag()
        self._calcute_pose(with_default_or_zero)
        return self._calculate_mean(error_std, penalize_weight) * ratios[0] \
               + self._calculate_var(error_std, penalize_weight) * ratios[1]

