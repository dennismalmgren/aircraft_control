from typing import Optional
from enum import IntEnum

import torch

class TankType(IntEnum):
    Unknown = 0
    Fuel = 1
    Oxidizer = 2

class GrainType(IntEnum):
    Unknown = 1
    Cylindrical = 2
    Endburning = 3
    Function = 4

class Tank:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.vXYZ = torch.zeros(self.size, 3, dtype=torch.float64, device=self.device)
        self.vXYZ_drain = torch.zeros(self.size, 3, dtype=torch.float64, device=self.device)
        self.Capacity = torch.tensor(0.00001, dtype=torch.float64, device=self.device).expand(*self.size, 1)
        self.Contents = torch.zeros(self.size, 1, dtype=torch.float64, device=self.device)

    def GetXYZ(self) -> torch.Tensor:
        return self.vXYZ_drain + (self.Contents / self.Capacity) * (self.vXYZ - self.vXYZ_drain)
    
    def GetContents(self) -> torch.Tensor:
        return self.Contents

    def reset_to_ic(self):
        pass
    #SetTemperature( InitialTemperature );
    #SetStandpipe ( InitialStandpipe );
    #SetContents ( InitialContents );
    #PctFull = 100.0*Contents/Capacity;
    #SetPriority( InitialPriority );
    #CalculateInertias();
