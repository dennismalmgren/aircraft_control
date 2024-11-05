from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.external_force import ExternalForce

class Element: #TODO
    def __init__(self):
        pass

class GasCellInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Pressure = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.Density = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.gravity = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)

class GasType(IntEnum):
    UnknownGas = 0
    Hydrogen = 1
    Helium = 2
    Air = 3

class GasCell:
    def __init__(self, inputs: GasCellInputs, 
                 element: Element,
                 num: int,
                 device: torch.device, batch_size: Optional[torch.Size] = None):
        self.Type: GasType = GasType.UnknownGas
        self.CellNum = num