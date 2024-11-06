from typing import Optional, List

import torch

from jsbsim_parallel.math.lagrange_multiplier import LagrangeMultiplier
from jsbsim_parallel.models.lgear import LGear, LGearInputs
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class GroundReactions(ModelBase):
    def __init__(self, 
                 path_provider: ModelPathProvider,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(path_provider, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.lGear: List[LGear] = [] #TODO, NO LISTS
        self.vForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.multipliers: List[LagrangeMultiplier]  = [] #TODO, NO LISTS
        self._in = LGearInputs(device, batch_size)
        self.DsCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        #fgsurface properties
        self.staticFFactor = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.rollingFFactor = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.maximumForce = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.bumpiness = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.isSolid = torch.zeros(*self.size, 1, dtype=torch.bool, device=device)
        self.pos = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

    def GetMultipliersList(self) -> List[LagrangeMultiplier]:
        return self.multipliers
    
    def GetWOW(self) -> torch.Tensor:
        if len(self.lGear) == 0:
            return torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        else:
            wow_status = torch.stack([
                gear.IsBogey() & gear.GetWOW() for gear in self.lGear
            ], dim=0)  # Shape: [num_gears, *batch_size]

            # Check if any gear in each batch has WOW
            return wow_status.any(dim=0)  # Shape: [*batch_size]

    #from fgsurface
    def resetValues(self):
        self.staticFFactor.fill_(1.0)
        self.rollingFFactor.fill_(1.0)
        self.maximumForce.fill_(torch.float64("inf"))
        self.bumpiness.zero_()
        self.isSolid.fill_(True)
        self.pos.zero_()

    def GetForces(self) -> torch.Tensor:
        return self.vForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vMoments
    
    def run(self, holding: bool) -> bool:
        if holding:
            return False

        return True

    def init_model(self) -> bool:
        return True