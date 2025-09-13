
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import commands_cfg

from .zero2small_command import ZeroSmallCommand


@configclass
class ZeroSmallCommandCfg(commands_cfg.UniformVelocityCommandCfg):
    class_type: type = ZeroSmallCommand

    small2zero_threshold_line: float = 0.2
    small2zero_threshold_angle: float = 0.1
