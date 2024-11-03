from typing import Optional
from enum import IntEnum

import torch

class EngineType(IntEnum):
    Unknown = 1
    Rocket = 2
    Piston = 3
    Turbine = 4
    Turboprop = 5
    Electric = 6
    
class EngineInputs:
    def __init__(self, *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        # Define the size for batch support
        size = batch_size if batch_size is not None else torch.Size([])

        # Scalar properties
        self.Pressure = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.PressureRatio = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Density = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.DensityRatio = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Soundspeed = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TotalPressure = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TAT_c = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Vt = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Vc = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.qbar = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.alpha = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.beta = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.H_agl = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TotalDeltaT = torch.zeros(*size, 1, dtype=torch.float64, device=device)

        # Vector properties (FGColumnVector3 equivalent)
        self.AeroUVW = torch.zeros(*size, 3, 1, dtype=torch.float64, device=device)
        self.AeroPQR = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PQRi = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # List properties for variable-length data
        self.ThrottleCmd = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.MixtureCmd = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.ThrottlePos = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.MixturePos = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.PropAdvance = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]

        self.PropFeather = [torch.zeros(*size, 1, dtype=torch.bool, device=device)]  
        
class Engine:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

    
    def reset_to_ic(self):
        pass