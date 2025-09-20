from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

from enum import Enum
class FeetStatus(Enum):
    NONE = 0
    Contact_Status = 1
    Air_Status = 2


class StepsBase(ManagerTermBase):

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        # 初始化StatusBase，加载配置和资产对象，并初始化统计数据缓冲区
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self.sensor_cfg = sensor_cfg
        self.command_name = cfg.params["command_name"]

        posecount = len(self.sensor_cfg.body_ids)
        self.feet_status_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.int8)

        self.air_steps_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.int8)

        self.contact_steps_buf = torch.zeros_like(self.air_steps_buf)
        self._init_buffers()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if not hasattr(self, "stand_flag"):
            ## stand_flag
            command: UniformVelocityCommand = self._env.command_manager.get_term(self.command_name)
            self.stand_flag = command.is_standing_env

        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return {}

        for _item in self.__dict__.items():
            if not _item[0].endswith("_buf"):
                continue

            _item[1][env_ids] = 0

        return {}

    def _calcute_step(self, diff: torch.Tensor, mask: torch.Tensor, buf_select: function) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        steps_buf, variance_buf, mean_buf = buf_select()

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - mean_buf
        work_mean_buf = mean_buf + delta0 / steps_buf

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - work_mean_buf
        work_variance_buf = (
            variance_buf * (steps_buf - 2)
            + delta0 * delta1
        ) / (steps_buf - 1)

        mean_buf[mask] = work_mean_buf[mask]
        variance_buf[mask] = work_variance_buf[mask]

        # 当episode刚开始时重置方差，防止数值异常
        new_mask = steps_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        variance_buf[new_mask] = 0

        if torch.isnan(mean_buf).float().sum() > 0 or torch.isinf(mean_buf).float().sum() > 0:
            print("is nan")

        if torch.isnan(variance_buf).float().sum() > 0 or torch.isinf(variance_buf).float().sum() > 0:
            print("is nan")

    def _calcute_status(self):
        contact = self.contact_sensor.data.current_contact_time[:, self.sensor_cfg.body_ids]
        in_contact = contact > 0.0

        init_mask = self.feet_status_buf == FeetStatus.NONE.value
        air_mask = self.feet_status_buf == FeetStatus.Air_Status.value
        contact_mask = self.feet_status_buf == FeetStatus.Contact_Status.value

        init_flags = torch.logical_and(in_contact, init_mask)
        update2contact = torch.logical_and(in_contact, air_mask)
        update2air = torch.logical_and(torch.logical_not(in_contact), contact_mask)

        self.feet_status_buf[init_flags] = FeetStatus.Contact_Status.value
        self.feet_status_buf[update2air] = FeetStatus.Air_Status.value
        self.feet_status_buf[update2contact] = FeetStatus.Contact_Status.value

        self.air_steps_buf[update2contact] += 1
        self.contact_steps_buf[update2air] += 1

        self.update2contact_mask = update2contact
        self.update2air_mask = update2air

        _mask = torch.logical_xor(update2contact[:, :1], update2contact[:, 1:])
        self.update_steps_mask = torch.repeat_interleave(_mask, 2, dim=-1)

        steps_buf = torch.sum(self.air_steps_buf, dim = -1, keepdim= True)
        self.steps_buf = torch.repeat_interleave(steps_buf, 2, dim=-1)


class StatusStep(StepsBase):

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):

        self.steptimes_flag = False
        if "steptimes_flag" in cfg.params:
            self.steptimes_flag = cfg.params["steptimes_flag"]

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg

        super().__init__(cfg, env)


    def _init_steptimes(self):
        if not self.steptimes_flag:
            return
        posecount = len(self.sensor_cfg.body_ids)

        self.air_variance_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.float)
        self.air_mean_buf = torch.zeros_like(self.air_variance_buf)

        self.contact_variance_buf = torch.zeros_like(self.air_variance_buf)
        self.contact_mean_buf = torch.zeros_like(self.air_variance_buf)

    def _init_feetpose(self):
        if self.sensor_cfg.body_names is None:
            return

        posecount = len(self.sensor_cfg.body_names)
        self.feetpose3d_variance_buf = torch.zeros((self.num_envs, posecount, 3),
                              device=self.device, dtype=torch.float)
        self.feetpose3d_mean_buf = torch.zeros_like(self.feetpose3d_variance_buf)

        # self.feetpose3d_last_buf = torch.zeros_like(self.feetpose3d_variance_buf)

    def _init_bodiespose(self):
        if self.asset_cfg.body_names is None:
            return
        posecount = len(self.asset_cfg.body_names)
        self.bodiespose3d_variance_buf = torch.zeros((self.num_envs, posecount, 3),
                              device=self.device, dtype=torch.float)
        self.bodiespose3d_mean_buf = torch.zeros_like(self.bodiespose3d_variance_buf)
        # self.bodiespose3d_last_buf = torch.zeros_like(self.bodiespose3d_variance_buf)

    def _init_jointspose(self):
        if self.asset_cfg.joint_names is None:
            return
        joint_count = len(self.asset_cfg.joint_names)
        self.jointspose_variance_buf = torch.zeros((self.num_envs, joint_count),
                              device=self.device, dtype=torch.float)
        self.jointspose_mean_buf = torch.zeros_like(self.jointspose_variance_buf)
        # self.jointspose_last_buf = torch.zeros_like(self.jointspose_mean_buf)

    def _init_buffers(self):
        self._init_steptimes()
        self._init_feetpose()
        self._init_bodiespose()
        self._init_jointspose()

    #########################################################
    def _calcute_steptimes(self):
        if not self.steptimes_flag:
            return

        air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        contact_time = self.contact_sensor.data.last_contact_time[:, self.sensor_cfg.body_ids]

        ##
        def select_air_buffer():
            return self.air_steps_buf, self.air_variance_buf, self.air_mean_buf
        ##
        def select_contact_buffer():
            return self.contact_steps_buf, self.contact_variance_buf, self.contact_mean_buf

        self._calcute_step(air_time, self.update2contact_mask, select_air_buffer)
        self._calcute_step(contact_time, self.update2air_mask, select_contact_buffer)


    def _calcute_feetpose(self):
        if not hasattr(self, "feetpose3d_variance_buf"):
            return

        pos_w = self.asset.data.body_pos_w[:, self.sensor_cfg.body_ids] - self.asset.data.root_pos_w[:, None, :]
        quat_w = torch.repeat_interleave(self.asset.data.root_quat_w[:, None, :], pos_w.shape[1], dim=1)
        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos_w)

        def select_buffer():
            steps_buf = torch.repeat_interleave(self.steps_buf[..., None], pos_b.shape[-1], dim=-1)
            steps_buf = torch.repeat_interleave(steps_buf, pos_b.shape[-2] // 2, dim=-2)
            return steps_buf, self.feetpose3d_variance_buf, self.feetpose3d_mean_buf

        mask = torch.repeat_interleave(self.update_steps_mask[..., None], pos_b.shape[-1], dim=-1)
        mask = torch.repeat_interleave(mask, pos_b.shape[-2] // 2, dim=-2)

        self._calcute_step(pos_b, mask, select_buffer)

    def _calcute_bodiespose(self):
        if not hasattr(self, "bodiespose3d_variance_buf"):
            return

        pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids] - self.asset.data.root_pos_w[:, None, :]
        quat_w = torch.repeat_interleave(self.asset.data.root_quat_w[:, None, :], pos_w.shape[1], dim=1)
        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos_w)

        def select_buffer():
            steps_buf = torch.repeat_interleave(self.steps_buf[..., None], pos_b.shape[-1], dim=-1)
            steps_buf = torch.repeat_interleave(steps_buf, pos_b.shape[-2] // 2, dim=-2)
            return steps_buf, self.bodiespose3d_variance_buf, self.bodiespose3d_mean_buf

        mask = torch.repeat_interleave(self.update_steps_mask[..., None], pos_b.shape[-1], dim=-1)
        mask = torch.repeat_interleave(mask, pos_b.shape[-2] // 2, dim=-2)

        self._calcute_step(pos_b, mask, select_buffer)

    def _calcute_jointspose(self):
        if not hasattr(self, "jointspose_variance_buf"):
            return

        joint_pos = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids] - self.asset.data.default_joint_pos[:, self.asset_cfg.joint_ids]

        def select_buffer():
            steps_buf = torch.repeat_interleave(self.steps_buf, joint_pos.shape[-1] // 2, dim=-1)
            return steps_buf, self.jointspose_variance_buf, self.jointspose_mean_buf

        mask = torch.repeat_interleave(self.update_steps_mask, joint_pos.shape[-1] // 2, dim=-1)
        self._calcute_step(joint_pos, mask, select_buffer)


    def __call__(self):
        self._calcute_status()
        self._calcute_feetpose()
        self._calcute_bodiespose()
        self._calcute_jointspose()
        self._calcute_steptimes()

    '''

    '''
    def statistics_feetids(self, asset_cfg):
        assert len(asset_cfg.body_ids) > 0 and len(self.sensor_cfg.body_ids) > 0
        ids = []
        for bid in asset_cfg.body_ids:
            id = self.sensor_cfg.body_ids.index(bid)
            ids.append(id)
        return ids

    def statistics_bodiesids(self, asset_cfg):
        assert len(asset_cfg.body_ids) > 0 and len(self.asset_cfg.body_ids) > 0
        ids = []
        for bid in asset_cfg.body_ids:
            id = self.asset_cfg.body_ids.index(bid)
            ids.append(id)
        return ids

    def statistics_jointsids(self, asset_cfg):
        assert len(asset_cfg.joint_ids) > 0 and len(self.asset_cfg.joint_ids) > 0
        ids = []
        for jid in asset_cfg.joint_ids:
            id = self.asset_cfg.joint_ids.index(jid)
            ids.append(id)
        return ids

