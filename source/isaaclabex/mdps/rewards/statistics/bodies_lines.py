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

class BodiesSymmetry(ManagerTermBase):
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
        posecount = len(self.asset_cfg.body_ids)
        self.episode_variance_buf = torch.zeros((self.num_envs, posecount, 3),
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
        self.episode_mean_buf += delta0 / episode_length_buf[:, None, None]

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None, None] - 1)

        # 当episode刚开始时重置方差，防止数值异常
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _calcute_lin_vel(self):
        lin_vel_w = self.asset.data.body_link_lin_vel_w[:, self.asset_cfg.body_ids]
        # 重复根链接的旋转四元数，使其与 pos 的维度匹配
        # 即将根链接的四元数应用到每个 body 上
        quat_w = torch.repeat_interleave(self.asset.data.root_quat_w[:, None, :], lin_vel_w.shape[1], dim=1)
        # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
        try:
            lin_vel_b = math_utils.quat_apply_inverse(quat_w, lin_vel_w) - self.asset.data.root_lin_vel_b[:, None, :]
        except:
            lin_vel_b = math_utils.quat_rotate_inverse(quat_w, lin_vel_w) - self.asset.data.root_lin_vel_b[:, None, :]

        self.current_lin_vel_b = lin_vel_b
        self._calculate_episode(lin_vel_b)

    def _calculate_zero(self, error, error_std):
        return torch.exp(- torch.square(error / error_std))

    def _calculate_penalize(self, error, error_std):
        offset = torch.abs(torch.abs(error) - error_std * 0.6)
        penalize = torch.clamp_max(0.5 - offset / (error_std* 0.6), max = 0)
        return torch.square(penalize)

    def _calculate_stand(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):
        # stand
        stand_reward = self._calculate_zero(self.current_lin_vel_b[self.stand_flag, :, 0], error_std) + \
            penalize_weight * self._calculate_penalize(self.current_lin_vel_b[self.stand_flag, :, 0], error_std)
        stand_reward = torch.sum(stand_reward, dim = -1)
        return stand_reward

    def _calculate_meanx(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        # mean
        episode_mean = self.episode_mean_buf[:, :, 0]

        reward_mean = self._calculate_zero(episode_mean, error_std) + \
            penalize_weight * self._calculate_penalize(episode_mean, error_std)

        if False:
            # current
            diff = self.current_lin_vel_b[:, ::2, 0] + self.current_lin_vel_b[:, 1::2, 0]

            reward_cur = self._calculate_zero(diff, error_std) + \
                penalize_weight * self._calculate_penalize(diff, error_std)
            # sum
            reward = (torch.sum(reward_mean, dim = -1) * 0.6 + torch.sum(reward_cur, dim = -1) * 0.4)
        else:
            reward = torch.sum(reward_mean, dim = -1)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)
        return reward

    def _calculate_varx(self, error_std: float = 0.06,
            penalize_weight: float = -0.25):

        episode_variance = self.episode_variance_buf[:, :, 0]
        episode_std = torch.sqrt(episode_variance)

        #
        error = episode_std[:, ::2] - episode_std[:, 1::2]
        reward = self._calculate_zero(error, error_std) + \
            penalize_weight * self._calculate_penalize(error, error_std)
        reward = torch.sum(reward, dim = -1)

        reward[self.stand_flag] = self._calculate_stand(error_std, penalize_weight)

        return reward

    def __call__(self,
            env: ManagerBasedRLEnv,
            command_name: str = "base_velocity",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            error_std: float = 0.06,
            penalize_weight: float = -0.25
        ) -> torch.Tensor:
        self._update_flag()
        self._calcute_lin_vel()
        return self._calculate_meanx(error_std, penalize_weight) + self._calculate_varx(error_std, penalize_weight)

