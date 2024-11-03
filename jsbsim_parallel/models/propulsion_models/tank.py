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

    
    def reset_to_ic(self):
        pass
    #SetTemperature( InitialTemperature );
    #SetStandpipe ( InitialStandpipe );
    #SetContents ( InitialContents );
    #PctFull = 100.0*Contents/Capacity;
    #SetPriority( InitialPriority );
    #CalculateInertias();
