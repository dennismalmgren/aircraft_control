from typing import Optional
from enum import Enum

import torch

class Propulsion:
#    FGFDMExec* Executive
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        
    def run(holding: bool) -> bool:
        pass