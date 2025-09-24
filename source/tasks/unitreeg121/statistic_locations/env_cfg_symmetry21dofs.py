from . import env_cfg
from .mdps import mdps_symmetry21dofs
from isaaclab.utils import configclass
from .mdps  import rewards


@configclass
class G1Cfg(env_cfg.G1PBRSCfg):
    commands: mdps_symmetry21dofs.CommandsCfg = mdps_symmetry21dofs.CommandsCfg()
    observations: mdps_symmetry21dofs.ObservationsCfg = mdps_symmetry21dofs.ObservationsCfg()


@configclass
class G1Cfg_PLAY(env_cfg.G1PBRSCfg_PLAY):
    commands: mdps_symmetry21dofs.CommandsCfg = mdps_symmetry21dofs.CommandsCfg()
    observations: mdps_symmetry21dofs.ObservationsCfg = mdps_symmetry21dofs.ObservationsCfg()


@configclass
class G1CfgV2(G1Cfg):
    rewards = rewards.NormalG21Cfg()

@configclass
class G1CfgV2_PLAY(G1Cfg_PLAY):
    rewards = rewards.NormalG21Cfg()

