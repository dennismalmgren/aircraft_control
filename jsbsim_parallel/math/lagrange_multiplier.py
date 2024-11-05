from typing import List, Optional
import torch

class LagrangeMultiplier:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.ForceJacobian = torch.zeros(self.size, 3, device=self.device)
        self.LeverArm = torch.zeros(self.size, 3, device=self.device)
        self.Min = torch.zeros(self.size, 1, device=self.device)
        self.Max = torch.zeros(self.size, 1, device=self.device)
        self.value = torch.zeros(self.size, 1, device=self.device)

