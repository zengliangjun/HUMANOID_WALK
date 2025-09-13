from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

def penalize_feet_orientation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1
    ):
    """
    Computes the penalty on feet orientation to make no x and y projected gravity.

    This function is adapted from _reward_feet_ori in legged_gym.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs) representing the computed penalty for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]

    vec = asset.data.GRAVITY_VEC_W[:, None, :].repeat((1, quat.shape[1], 1))
    gravity = math_utils.quat_apply_inverse(quat, vec)

    error = torch.sum(torch.square(gravity[..., :2]), dim=-1) ** 0.5  ## n x num_feet

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold  ## n x num_feet
    error[feet_contact] = 0.0

    return torch.sum(error, dim=-1)

