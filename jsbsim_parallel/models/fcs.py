from typing import Optional, List, Union
from enum import IntEnum

import torch

from jsbsim_parallel.models.lgear import BrakeGroup

class OutputForm:
    Rad = 0
    Deg = 1
    Norm = 2
    Mag = 3
    NForms = 4

class FCS:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        #channelrate 1
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.DaCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DeCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DrCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DfCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DsbCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DspCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.PTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.YTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.RTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.GearCmd = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        self.GearPos = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        #vectors
        self.ThrottleCmd: List[torch.Tensor] = []
        self.ThrottlePos = []
        self.MixtureCmd = []
        self.MixturePos = []
        self.PropAdvanceCmd: List[torch.Tensor] = []
        self.PropAdvance: List[torch.Tensor] = []
        self.PropFeatherCmd: List[torch.Tensor] = []
        self.PropFeather: List[torch.Tensor] = []
        self.BrakePos: List[torch.Tensor] = [torch.zeros(*self.size, 1, dtype=torch.float64, device=device) for _ in range(BrakeGroup.NumBrakeGroups)]
        #brakepos resize..
        self.TailhookPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.WingFoldPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        #bind
        self.DePos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DaLPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DaRPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DrPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DfPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DsbPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DspPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        
    def GetPropFeather(self) -> List[torch.Tensor]:
        return self.PropFeather
    
    def GetPropAdvance(self) -> List[torch.Tensor]:
        return self.PropAdvance
    
    def GetMixtureCmd(self) -> List[torch.Tensor]:
        return self.MixtureCmd
    
    def GetMixturePos(self) -> List[torch.Tensor]:
        return self.MixturePos
    
    def GetThrottleCmd(self) -> List[torch.Tensor]:
        return self.ThrottleCmd
        
    def GetThrottlePos(self, engine: Optional[int] = None) -> Union[List[torch.Tensor], torch.Tensor]:
        if engine is None:
            return self.ThrottlePos
        else:
            return self.ThrottlePos[engine]

    def GetBrakePos(self) -> List[torch.Tensor]:
        return self.BrakePos
    
    def GetGearPos(self) -> torch.Tensor:
        return self.GearPos
    
    def run(self, holding: bool) -> bool:
        if holding:
            return True

        return False

    def init_model(self) -> bool:

        #reset to IC.
        return True

