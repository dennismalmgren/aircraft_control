from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class AircraftInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Use batch_size for initialization if provided, else default to a single instance.
        size = batch_size if batch_size is not None else torch.Size([])
        self.AeroForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PropForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GroundForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.ExternalForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.BuoyantForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.AeroMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PropMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GroundMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.ExternalMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.BuoyantMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)



class Aircraft(ModelBase):
    def __init__(self, 
                 path_provider: ModelPathProvider,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size if batch_size is not None else torch.Size([])
        super().__init__(path_provider, device=device, batch_size=batch_size)

        self.path_provider = path_provider
        self._in = AircraftInputs(device, batch_size)

        self.vMoments = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vForces = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZrp = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZvrp = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZep = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vDXYZcg = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)

        self.WingArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.WingSpan = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.cbar = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.WingIncidence = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.HTailArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.VTailArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.HTailArm = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.VTailArm = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.lbarh = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.lbarv = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.vbarh = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.vbarv = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        if len(self.batch_size) == 0:
            self.aircraft_names = ["FGAircraft"]
        else:
            self.aircraft_names = ["FGAircraft" for _ in range(len(self.batch_size[0]))]

    def SetAircraftName(self, name: str):
        if len(self.batch_size) == 0:
            self.aircraft_names = [name]
        else:
            self.aircraft_names = [name for _ in range(len(self.batch_size[0]))]
    
    def Load(self, element: Element) -> bool:
        if not super().Upload(element, True):
            return False

        return True
    
    def GetForces(self) -> torch.Tensor:
        return self.vForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vMoments
    
    def GetXYZep(self) -> torch.Tensor:
        return self.vXYZep
    
    def GetXYZvrp(self) -> torch.Tensor:
        return self.vXYZvrp
    
    def GetXYZrp(self) -> torch.Tensor:
        return self.vXYZrp
    
    def run(self, holding:bool) -> bool:
        if holding:
            return False
        
        return True

    def init_model(self) -> bool:
        return True