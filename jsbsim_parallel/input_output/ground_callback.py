from typing import Optional

import torch

class GroundCallback:
    def __init__(self, semi_major: torch.Tensor, semi_minor: torch.Tensor, *, device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size
        self.time = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.a = semi_major
        self.b = semi_minor
        self.terrain_elevation = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)

    