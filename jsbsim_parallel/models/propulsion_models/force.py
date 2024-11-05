from typing import Optional
from enum import IntEnum

import torch


from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.models.model_base import ModelBase

class TransformType:
    NoTransform = 0
    WindBody = 1
    LocalBody = 2
    InertialBody = 3
    Custom = 4

class Force(ModelBase):
    def __init__(self, mass_balance: MassBalance, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(self, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.mass_balance = mass_balance
        self.vFn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vMn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vOrient = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.ttype: TransformType = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device) #TransformType enum
        self.vXYZn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vActingXYZn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.mT = torch.eye(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        
        self.vFb = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vM = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        
    def SetTransformType(self, ttype: torch.Tensor):
        self.ttype = ttype

    def SetLocation(self, vv: torch.Tensor):
        self.vXYZn = vv
        self.SetActingLocation(vv)
        
    def SetActingLocation(self, vv: torch.Tensor):
        self.vActingXYZn = vv