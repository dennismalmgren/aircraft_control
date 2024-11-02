from typing import Optional
from enum import IntEnum

import torch

from jsbsim_parallel.models.lgear import BrakeGroup

class OutputForm:
    Rad = 0
    Deg = 1
    Norm = 2
    Mag = 3
    NForms = 4

class FGFCS:
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
        #brakepos resize..
        self.BrakePos = torch.zeros(*self.size, BrakeGroup.NumBrakeGroups, dtype=torch.float64, device=device)
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
        

    def run(holding: bool) -> bool:
        if holding:
            return True

        return False

    def init_model() -> bool:
        
        #reset to IC.
        return True

