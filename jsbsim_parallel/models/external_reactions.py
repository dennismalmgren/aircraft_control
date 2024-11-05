from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.external_force import ExternalForce

class ExternalReactions:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.Forces: List[ExternalForce] = []
        self.vTotalForces = torch.zeros(*batch_size, 3, dtype=torch.float64, device=device)
        self.vTotalMoments = torch.zeros(*batch_size, 3, dtype=torch.float64, device=device)

    def run(self, holding: bool) -> bool:
        if holding:
            return False

        return True

    def init_model(self) -> bool:
        self.vTotalForces.zero_()
        self.vTotalMoments.zero_()
        return True