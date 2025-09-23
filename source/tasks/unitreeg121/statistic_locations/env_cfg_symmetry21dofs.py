from . import env_cfg
from .mdps import mdps_symmetry21dofs
from isaaclab.utils import configclass

@configclass
class G1Cfg(env_cfg.G1PBRSCfgRNN):
    commands: mdps_symmetry21dofs.CommandsCfg = mdps_symmetry21dofs.CommandsCfg()
    observations: mdps_symmetry21dofs.ObservationsCfg = mdps_symmetry21dofs.ObservationsCfg()


@configclass
class G1Cfg_PLAY(env_cfg.G1PBRSCfgRNN_PLAY):
    commands: mdps_symmetry21dofs.CommandsCfg = mdps_symmetry21dofs.CommandsCfg()
    observations: mdps_symmetry21dofs.ObservationsCfg = mdps_symmetry21dofs.ObservationsCfg()


