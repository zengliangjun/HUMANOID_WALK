
from dataclasses import MISSING
from typing import Literal
import torch

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg

from abc import ABC, abstractmethod

class SymmetryClass(ABC):

    @abstractmethod
    def mirror_policy(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def mirror_critic(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def mirror_action(self, actions: torch.Tensor) -> torch.Tensor:
        pass

@configclass
class SymmetryActorCriticCfg(RslRlPpoActorCriticCfg):
    '''
    rsl_rlex.symmetry.modules.modules:SymmetryActorCritic
    '''
    class_name: str = "SymmetryActorCritic"

    symmetry_objs: SymmetryClass = MISSING


@configclass
class SymmetryRecurrentCfg(RslRlPpoActorCriticRecurrentCfg):

    class_name: str = "SymmetryActorCriticRecurrent"

    symmetry_objs: SymmetryClass = MISSING




