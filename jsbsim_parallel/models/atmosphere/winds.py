from typing import Optional
from enum import IntEnum

import torch

from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class TurbulenceType(IntEnum):
    NoTurbulence: int = 0
    Standard: int = 1
    Culp: int = 2
    Milspec: int = 3
    Tustin: int = 4

class WindsInputs:
    def __init__(self, batch_size: Optional[torch.Size] = None, device: torch.device = torch.device("cpu")):
        # Define the size as a batch_size if provided, otherwise scalar
        size = batch_size if batch_size is not None else torch.Size([])

        # Initialize each attribute as a tensor with the appropriate size and device
        self.V = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.wingspan = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.DistanceAGL = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.AltitudeASL = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.longitude = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.latitude = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.planetRadius = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        
        # FGMatrix33 equivalent (3x3 matrix)
        self.Tl2b = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.Tw2b = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.totalDeltaT = torch.zeros(*size, 1, dtype=torch.float64, device=device)

class Winds(ModelBase):
    def __init__(self, 
                 path_provider: ModelPathProvider,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(path_provider, device=device, batch_size=batch_size)
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self._turbulence_Type = TurbulenceType.NoTurbulence

        # double MagnitudedAccelDt, MagnitudeAccel, Magnitude, TurbDirection;
        # double TurbGain;
        # double TurbRate;
        # double Rhythmicity;
        # double wind_from_clockwise;
        # double spike, target_time, strength;
        # FGColumnVector3 vTurbulenceGrad;
        # FGColumnVector3 vBodyTurbGrad;
        # FGColumnVector3 vTurbPQR;

        self.MagnitudedAccelDt = torch.zeros(*self.size, 1, dtype=torch.float64, device=device) 
        self.MagnitudeAccel = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Magnitude = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TurbDirection = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TurbGain = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TurbRate = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Rhythmicity = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.wind_from_clockwise = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.spike = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.target_time = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.strength = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.vTurbulenceGrad = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vBodyTurbGrad = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vTurbPQR = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

        self.vTotalWindNED = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vGustNED = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vCosineGust = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vWindNED = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vBurstGust = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vTurbulenceNED = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.psiw = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self._in = WindsInputs(device=device, batch_size=self.size)

    def GetTotalWindNED(self) -> torch.Tensor:
        return self.vTotalWindNED
    
    def GetTurbPQR(self) -> torch.Tensor:
        return self.vTurbPQR
    
    def run(self, holding: bool) -> bool:
        if holding:
            return True
        
        return False

    def init_model(self) -> bool:
        #reset to IC.
        return True