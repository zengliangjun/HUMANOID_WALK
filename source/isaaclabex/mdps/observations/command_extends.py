
from isaaclab.envs import ManagerBasedRLEnv
from isaaclabex.mdps.commands.zero2small_command import SymmetryCommand

import torch

def symmetry_flags(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: SymmetryCommand  =  env.command_manager.get_term(command_name)
    assert isinstance(term, SymmetryCommand)
    return term.symmetry_flags[:, None]

