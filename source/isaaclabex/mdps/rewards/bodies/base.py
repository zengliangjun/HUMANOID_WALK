from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils import math
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def penalize_base_height(env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes deviation of the robot's base height from the target height using L2 squared error.

    Args:
        env (ManagerBasedRLEnv): Environment instance for simulation.
        target_height (float): Desired target height for the robot's base.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: L2 squared penalty of the base height deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the individual feet heights and select the minimum (ground contact)
    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height = torch.min(feet_heights, dim=-1)[0]
    # Compute body height as absolute difference between base height and lowest foot height
    body_height = torch.abs(asset.data.root_pos_w[:, 2] - feet_height)

    # Return L2 squared penalty based on the difference from target height
    return torch.square(body_height - target_height)

def reward_penalize_base_height(env: ManagerBasedRLEnv,
    target_height: float,
    error_std: float = 0.025,
    penalize_weight: float = -3.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes deviation of the robot's base height from the target height using L2 squared error.

    Args:
        env (ManagerBasedRLEnv): Environment instance for simulation.
        target_height (float): Desired target height for the robot's base.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: L2 squared penalty of the base height deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the individual feet heights and select the minimum (ground contact)
    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height = torch.min(feet_heights, dim=-1)[0]
    # Compute body height as absolute difference between base height and lowest foot height
    base_height = torch.abs(asset.data.root_pos_w[:, 2] - feet_height)

    command: UniformVelocityCommand = env.command_manager.get_term(command_name)
    stand_flag = command.is_standing_env

    base_height[stand_flag] = torch.abs(base_height[stand_flag] - target_height)
    walk_flag = torch.logical_not(stand_flag)
    base_height[walk_flag] = torch.abs(base_height[walk_flag] - (target_height - error_std))

    # Return L2 squared penalty based on the difference from target height
    return torch.exp(- base_height / error_std) + penalize_weight * (base_height / error_std)

def reward_penalize_height_upper(env: ManagerBasedRLEnv,
    target_height: float,
    error_std: float = 0.025,
    penalize_weight: float = -3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes deviation of the robot's base height from the target height using L2 squared error.

    Args:
        env (ManagerBasedRLEnv): Environment instance for simulation.
        target_height (float): Desired target height for the robot's base.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: L2 squared penalty of the base height deviation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the individual feet heights and select the minimum (ground contact)
    upper_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    # Compute body height as absolute difference between base height and lowest foot height
    height = torch.abs(asset.data.root_pos_w[:, 2] - upper_height)
    error = torch.abs(height - target_height)

    # Return L2 squared penalty based on the difference from target height
    return torch.exp(- error / error_std) + penalize_weight * (error / error_std)

def reward_orientation_euler_gravity_b(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation
    from the desired base orientation using the base euler angles and the projected gravity vector.

    Args:
        env (ManagerBasedRLEnv): The environment instance, which contains the simulation scene
            and provides access to the assets and their states.
        asset_cfg (SceneEntityCfg): Configuration for the asset whose orientation is being evaluated.
            Defaults to a configuration with the name "robot".

    Returns:
        torch.Tensor: The calculated reward value, which is a combination of penalties for
        deviations in Euler angles and the projected gravity vector.
    """
    # Extract the asset from the environment using the provided configuration
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the root quaternion of the asset
    quat = asset.data.root_quat_w

    # Convert the quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = math.euler_xyz_from_quat(quat)

    # Calculate mismatch based on Euler angles (penalizes deviation from flat orientation)
    euler_mismatch = torch.exp(-(torch.abs(roll) + torch.abs(pitch)) * 10)

    # Calculate mismatch based on the projected gravity vector (penalizes deviation from vertical alignment)
    gravity_mismatch = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20)

    # Combine the two mismatch values into a single reward (average of both components)
    return (euler_mismatch + gravity_mismatch) / 2.

