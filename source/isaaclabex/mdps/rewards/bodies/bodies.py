from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reward_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    max_threshold: float = 0.2,
    min_threshold: float = 0.3):
    """
    Calculates the reward based on the distance between the feet.
    Penalizes if the feet are too close or too far from each other.

    Parameters:
        env (ManagerBasedRLEnv): The environment containing scene and asset data.
        asset_cfg (SceneEntityCfg): Configuration for the scene entity including name and body identifiers.
        max_threshold (float): Upper distance threshold; distances above are penalized. Default is 0.2.
        min_threshold (float): Lower distance threshold; distances below are penalized. Default is 0.3.

    Returns:
        torch.Tensor: Computed reward tensor based on the feet distance.
    """
    # Retrieve the articulation asset from the scene using the asset configuration name.
    asset: Articulation = env.scene[asset_cfg.name]

    # Extract the 2D positions (x, y) of the bodies based on provided body_ids.
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    # Compute Euclidean distance between paired body positions (assumes feet are ordered alternately).
    pos_dist = torch.norm(pos[:, ::2] - pos[:, 1::2], dim=1)

    # Compute a penalty for distances below the minimum threshold.
    d_min = torch.clamp(pos_dist - min_threshold, -0.5, 0.)
    # Compute a penalty for distances above the maximum threshold.
    d_max = torch.clamp(pos_dist - max_threshold, 0, 0.5)

    # Calculate and return the final reward as the average penalty over the two conditions.
    return torch.sum((torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, dim=-1)

def reward_distance_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    max_threshold: float = 0.2,
    min_threshold: float = 0.3):
    """
    Calculates the reward based on the distance between the feet.
    Penalizes if the feet are too close or too far from each other.

    Parameters:
        env (ManagerBasedRLEnv): The environment containing scene and asset data.
        asset_cfg (SceneEntityCfg): Configuration for the scene entity including name and body identifiers.
        max_threshold (float): Upper distance threshold; distances above are penalized. Default is 0.2.
        min_threshold (float): Lower distance threshold; distances below are penalized. Default is 0.3.

    Returns:
        torch.Tensor: Computed reward tensor based on the feet distance.
    """
    # Retrieve the articulation asset from the scene using the asset configuration name.
    asset: Articulation = env.scene[asset_cfg.name]

    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    # 重复根链接的旋转四元数，使其与 pos 的维度匹配
    # 即将根链接的四元数应用到每个 body 上
    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

    # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
    try:
        pos_b = math_utils.quat_apply_inverse(quat_w, pos)
    except:
        pos_b = math_utils.quat_rotate_inverse(quat_w, pos)


    # Extract the 2D positions (x, y) of the bodies based on provided body_ids.
    # pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]

    # Compute Euclidean distance between paired body positions (assumes feet are ordered alternately).
    pos_dist = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1])

    # Compute a penalty for distances below the minimum threshold.
    d_min = torch.clamp(pos_dist - min_threshold, -0.5, 0.)
    # Compute a penalty for distances above the maximum threshold.
    d_max = torch.clamp(pos_dist - max_threshold, 0, 0.5)

    # Calculate and return the final reward as the average penalty over the two conditions.
    return torch.sum((torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, dim=-1)

def reward_width(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    target_width: float = 0.2):
    """
    计算基于机器人身体宽度偏差的 reward。

    参数:
        env (ManagerBasedRLEnv): 环境对象，包含场景和资产数据。
        asset_cfg (SceneEntityCfg): 包含资产名称和需要计算的 body_id 的配置。
        target_width (float): 希望达到的目标宽度（默认值为 0.2）。

    返回:
        torch.Tensor: 基于宽度偏差计算出的 reward 张量（求平均值）。
    """

    # 从环境中获取指定资产对象（机器人）
    asset: Articulation = env.scene[asset_cfg.name]

    # 提取机器人中指定 body_ids 的3D位置（世界坐标）
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    # 重复根链接的旋转四元数，使其与 pos 的维度匹配
    # 即将根链接的四元数应用到每个 body 上
    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

    # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
    try:
        pos_b = math_utils.quat_apply_inverse(quat_w, pos)
    except:
        pos_b = math_utils.quat_rotate_inverse(quat_w, pos)


    # width 计算：
    # 1. 从基座坐标系 pos_b 中提取 y 轴坐标（索引 1 表示 y 分量）
    # 2. 假定 body 按照成对顺序排列，其中偶数索引和奇数索引代表一对
    # 3. 计算每一对中两个身体的 y 坐标差的绝对值，得到当前宽度
    # 4. 减去目标宽度 target_width，得到超出或不足目标的偏差
    width = (pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1] - target_width) * 100
    width[width < 0] = torch.square(width[width < 0] * 2)  # 确保负值被置为 0，避免不必要的惩罚
    width[width < 4] = 0

    # 将偏差放大（乘以 100），计算其指数惩罚，最后求所有对的平均值作为最终 reward
    return torch.mean(torch.exp(-width / 6), dim=-1)


def penalize_width(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    target_width: float = 0.2,
    target_height: float = 0.78,
    center_velocity: float = 0.4):
    """优化后的脚部宽度惩罚函数，包含动态阈值和连续惩罚

    参数:
        env: 环境对象
        asset_cfg: 资产配置
        target_width: 基础目标宽度

    返回:
        基于动态宽度阈值的连续惩罚值
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    # 转换到基座坐标系
    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)
    try:
        pos_b = math_utils.quat_apply_inverse(quat_w, pos)
    except:
        pos_b = math_utils.quat_rotate_inverse(quat_w, pos)


    # 计算当前脚部宽度(成对计算)
    current_width = pos_b[:, 0::2, 1] - pos_b[:, 1::2, 1]

    # 动态阈值计算(基于高度和速度)
    height = asset.data.root_link_pos_w[:, 2:]
    velocity = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1, keepdim= True)
    width_factor = 0.8 + 0.2 * torch.sigmoid(height*2 - target_height) + 0.4 * torch.sigmoid(velocity*5 - center_velocity*5)
    dynamic_width = target_width * width_factor

    # 连续惩罚计算
    width_ratio = current_width / dynamic_width
    penalty = torch.sigmoid((0.5 - width_ratio) * 10)

    # 安全宽度保护(绝对最小值)
    # 单位说明：min_safe_width和current_width均为米(m)
    min_safe_width = target_width * 0.6  # 0.6倍目标宽度(单位:m)
    unsafe_penalty = torch.where(current_width < min_safe_width,
                               (min_safe_width - current_width) * 20,  # 10为放大系数(无单位)
                               0)

    # 综合惩罚
    total_penalty = penalty + unsafe_penalty
    return torch.mean(total_penalty, dim=-1)


class Stability(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = self._env.scene[self.asset_cfg.name]


    def reset(self, env_ids: Sequence[int] | None = None):
        # 如果指定环境列表为空则不执行重置
        if len(env_ids) == 0:
            return

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:

        # 提取机器人中指定 body_ids 的3D位置（世界坐标）
        pos = self.asset.data.body_pos_w[:, asset_cfg.body_ids]
        # 重复根链接的旋转四元数，使其与 pos 的维度匹配
        # 即将根链接的四元数应用到每个 body 上
        quat_w = torch.repeat_interleave(self.asset.data.root_link_quat_w[:, None, :], pos.shape[1], dim=1)

        # 对提取的位置进行逆旋转转换，将世界坐标系位置转换到机器人基座坐标系
        try:
            pos_b = math_utils.quat_apply_inverse(quat_w, pos)
            pos = torch.mean(pos_b[:, :, :2], dim=1)
            com_b = math_utils.quat_apply_inverse(self.asset.data.root_link_quat_w, self.asset.data.root_com_pos_w)
        except:
            pos_b = math_utils.quat_rotate_inverse(quat_w, pos)
            pos = torch.mean(pos_b[:, :, :2], dim=1)
            com_b = math_utils.quat_rotate_inverse(self.asset.data.root_link_quat_w, self.asset.data.root_com_pos_w)

        # 双足支撑稳定性计算 (只有两个支撑点)
        left_ankle = pos_b[:, 0, :2]  # 左脚踝位置(x,y)
        right_ankle = pos_b[:, 1, :2]  # 右脚踝位置(x,y)

        # 1. 计算支撑线向量和法向量
        support_vec = right_ankle - left_ankle
        support_len = torch.linalg.norm(support_vec, dim=1)
        support_dir = support_vec / (support_len.unsqueeze(1) + 1e-6)
        normal_vec = torch.stack([-support_dir[:, 1], support_dir[:, 0]], dim=1)

        # 2. 计算质心在支撑线上的投影
        com_proj = com_b[:, :2] - left_ankle
        com_dist = torch.sum(com_proj * normal_vec, dim=1)  # 垂直距离
        com_along = torch.sum(com_proj * support_dir, dim=1)  # 沿线距离

        # 3. 稳定性判断
        # 投影是否在支撑线范围内 (考虑10%的裕度)
        margin = support_len * 0.1
        stable = (com_along >= -margin) & (com_along <= (support_len + margin))

        # 4. 计算稳定性奖励
        # 距离越小越稳定，使用指数衰减函数
        dist_reward = torch.exp(-2.0 * torch.abs(com_dist))
        range_reward = torch.where(stable, 1.0, 0.2)
        total_reward = dist_reward * range_reward

        return total_reward

def penalize_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg):
    """
    Penalizes collisions detected by the contact sensor.

    Args:
        env (ManagerBasedRLEnv): Environment instance containing simulation scene and sensor data.
        sensor_cfg (SceneEntityCfg): Configuration for the sensor, including the sensor name and body part indices to monitor.

    Returns:
        torch.Tensor: A tensor representing the collision penalty, computed by summing contacts exceeding a force threshold.
    """
    # Retrieve the contact sensor instance using the provided sensor configuration
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract the net force values for the specified body parts from the sensor data
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    # Identify collisions where the force norm exceeds 0.1
    collision = torch.norm(forces, dim=-1) > 0.1
    # Sum the collision flags across the body parts to compute the overall penalty per environment sample
    return torch.sum(collision, dim=1)
# Inline and parameter annotations already improve code readability.

def rew_pitch_total2zero(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract the net force values for the specified body parts from the sensor data
    iscontact = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1


    # Retrieve the articulation asset.
    asset: Articulation = env.scene[asset_cfg.name]
    # Extract current and default joint positions.
    pose = asset.data.joint_pos[:, asset_cfg.joint_ids]

    total0 = torch.abs(torch.sum(pose[:, ::2], dim=-1))
    total1 = torch.abs(torch.sum(pose[:, 1::2], dim=-1))

    reward0 = torch.exp(-total0)
    reward1 = torch.exp(-total1)

    reward0[iscontact[:, 0]] = 0
    reward1[iscontact[:, 1]] = 0

    return (reward0 + reward1) / 2
