from __future__ import annotations
from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

def penalize_feet_forces_z(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_forces: float = 400) -> torch.Tensor:
    """
    Rewards high vertical contact forces recorded by the feet sensor.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which reward begins.
        max_forces (float): Maximum force value to be considered for reward clipping.

    Returns:
        torch.Tensor: The reward value, computed as the maximum clamped vertical force across specified body parts.
    """
    # Retrieve the contact sensor instance using the sensor configuration.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Extract vertical (z-axis) force values for specified body parts.
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Compute reward: clamp the difference between forces and threshold.
    _reward = torch.clamp(forces_z - threshold, min=0, max=max_forces)
    # Return the maximum reward among all monitored body parts.
    _reward = torch.max(_reward, dim=1)[0]
    return _reward

def penalize_feet_forces(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 500, max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Compute the norm of forces for specified body parts.
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces)
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def penalty_feet_airborne(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg, threshold: float = 1e-3):
    """
    Penalizes the robot when it is airborne, i.e., when feet lose contact with the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Contact time threshold below which a foot is considered airborne.

    Returns:
        torch.Tensor: A binary tensor (float) where 1.0 indicates the robot is airborne and 0.0 otherwise.

    Raises:
        RuntimeError: If the sensor's air time tracking is not activated.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Ensure that track_air_time is enabled in the sensor configuration.
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # Get the last contact times for monitored body parts.
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # Determine if any contact time is below the threshold; indicates airborne condition.
    return torch.any(last_contact_time < threshold, dim=1).float()

# The reward and penalty functions include detailed parameter and inline documentation.
def penalize_both_feet_in_air(env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg, threshold: float = 1) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_in_air = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] <= threshold

    return torch.all(feet_in_air, dim=1).float()



#######################################################

def penalize_feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    contacts_threshold: float = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalizes feet sliding on the ground by measuring the linear velocity of the feet
    when they are in contact with the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor; provides sensor name and body IDs.
        contacts_threshold (float): Threshold for contact forces to determine ground contact.
        asset_cfg (SceneEntityCfg): Configuration for the asset, defaulting to "robot".

    Returns:
        torch.Tensor: A penalty value computed from the linear velocity (norm) of the feet while in contact.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine if the contact force exceeds the threshold over history.
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contacts_threshold
    # Retrieve the asset instance.
    asset = env.scene[asset_cfg.name]
    # Extract linear velocity of the feet (only consider x and y components).
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # Compute norm of velocity for each foot and apply penalty only when contact is detected.
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def penalty_feet_stumble(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg):
    """
    Penalizes lateral forces on the feet that could indicate stumbling.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.

    Returns:
        torch.Tensor: A binary tensor (float) where 1.0 indicates a stumbling event and 0.0 otherwise.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get vertical forces (z-axis) for relevant body parts.
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Compute lateral forces from the x and y components.
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize if any lateral force exceeds the vertical force; return as binary (float) indicator.
    return torch.any(forces_xy > forces_z, dim=1).float()

#######################################################

class penalize_max_feet_height_before_contact(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        size = len(cfg.params["sensor_cfg"].body_ids)
        self._feet_max_height_in_air = torch.zeros((self.num_envs, size), dtype = torch.float32, device= self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._feet_max_height_in_air[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        target_height: float = 0.25) -> torch.Tensor:

        asset: RigidObject = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

        first_contact = contact_sensor.compute_first_contact(self._env.step_dt)[:, sensor_cfg.body_ids]

        feet_in_air = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] <= 1.0

        feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

        self._feet_max_height_in_air = torch.max(self._feet_max_height_in_air, feet_height)
        feet_max_height = torch.sum(
            (torch.clamp_min(target_height - self._feet_max_height_in_air, 0))
            * first_contact,
            dim=1,
        )  # reward only on first contact with the ground
        self._feet_max_height_in_air *= feet_in_air
        return feet_max_height


def penalize_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08) -> torch.Tensor:
    """
    Rewards the swinging feet for clearing a specified height off the ground.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        asset_cfg (SceneEntityCfg): Configuration for the asset; used to access the corresponding RigidObject.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor, including sensor name and monitored body IDs.
        target_height (float): The desired clearance height for the feet.

    Returns:
        torch.Tensor: A penalty value computed as the squared error between the feet elevation and target height,
                      applied only when no contact is detected.
    """
    # Retrieve the asset object representing the feet.
    asset: RigidObject = env.scene[asset_cfg.name]
    # Retrieve the contact sensor instance using sensor configuration.
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    # Determine if there has been significant contact over history (force threshold > 1.0).
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # Compute squared error between actual height and the target height.
    # The error is only considered if no contact is detected (~contacts).
    pos_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height) * ~contacts
    # Sum error across the body parts.
    return torch.sum(pos_error, dim=-1)
