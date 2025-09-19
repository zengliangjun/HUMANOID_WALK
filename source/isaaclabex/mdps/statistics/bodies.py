from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

class StatusBase(ManagerTermBase):

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        # 初始化StatusBase，加载配置和资产对象，并初始化统计数据缓冲区
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg
        self.command_name = cfg.params["command_name"]

        self._init_buffers()
        # 初始化标志位
        self.stand_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.zero_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def statistics_ids(self, asset_cfg):
        ids = []
        for bid in asset_cfg.body_ids:
            id = self.asset_cfg.body_ids.index(bid)
            ids.append(id)
        return ids

    def _episode_length(self) -> torch.Tensor:
        # 获取episode长度，当cfg中设置了截断时进行最大值截断
        if -1 == self.cfg.episode_truncation:
            return self._env.episode_length_buf
        else:
            return torch.clamp_max(self._env.episode_length_buf, self.cfg.episode_truncation)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        episode_length_buf = self._episode_length()

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

    def _update_flag(self):
        command: UniformVelocityCommand = self._env.command_manager.get_term(self.command_name)
        self.stand_flag[...] = command.is_standing_env
        self.zero_flag[...] = self._env.episode_length_buf <= 1

class StatusPose3d(StatusBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _init_buffers(self):
        # 初始化足接触统计的均值与方差缓冲区 (一维数据)
        posecount = len(self.asset_cfg.body_ids)
        self.episode_variance_buf = torch.zeros((self.num_envs, posecount, 3),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        # 清空所有足接触统计缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return {}

    def __call__(self):
        """执行统计计算"""
        # 提取机器人中指定 body_ids 的3D位置（世界坐标）
        pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids] - self.asset.data.root_pos_w[:, None, :]

        # 重复根链接的旋转四元数，使其与 pos 的维度匹配
        # 即将根链接的四元数应用到每个 body 上
        quat_w = torch.repeat_interleave(self.asset.data.root_quat_w[:, None, :], pos_w.shape[1], dim=1)
        # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos_w)


        self._calculate_episode(pos_b)
        self._update_flag()
