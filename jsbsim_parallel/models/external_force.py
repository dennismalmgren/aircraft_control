from typing import Optional, Dict
from enum import IntEnum

import torch

class ExternalForce:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        