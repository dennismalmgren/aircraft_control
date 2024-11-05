from typing import Optional, Dict, List
from enum import IntEnum

import torch


class BuoyantForces:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.vTotalForces = torch.zeros(*batch_size, 3, dtype=torch.float64, device=device)
        self.vTotalMoments = torch.zeros(*batch_size, 3, dtype=torch.float64, device=device)
        