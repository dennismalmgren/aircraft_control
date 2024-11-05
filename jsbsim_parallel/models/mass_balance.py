from typing import Optional
import torch

from jsbsim_parallel.models.propagate import Propagate 
from jsbsim_parallel.models.model_base import ModelBase

class MassBalance(ModelBase):
    def __init__(self, propagate: Propagate, *, device, batch_size: Optional[torch.Size] = None):
        super().__init__(device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.propagate = propagate
        self.weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.empty_weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.mass = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        
        self.vbaseXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.baseJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.J = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.Jinv = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.pmJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

    def GetXYZcg(self):
        return self.vXYZcg
    
    def GetEmptyWeight(self):
        return self.empty_weight
    
    def run(holding: bool) -> bool:
        if (holding):
            return True
        return False

    def init_model(self) -> bool:
        if not super().init_model():
            return False
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        
        return True