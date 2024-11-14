from typing import Optional
from enum import IntEnum

import torch


from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.models.model_base import ModelBase, EulerAngles
from jsbsim_parallel.input_output.simulator_service import SimulatorService

class TransformType:
    NoTransform = 0
    WindBody = 1
    LocalBody = 2
    InertialBody = 3
    Custom = 4

class Force(ModelBase):
    def __init__(self, mass_balance: MassBalance, 
                 simulator_service: SimulatorService,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(simulator_service,
                         device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.mass_balance = mass_balance
        self.vFn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vMn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vOrient = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.ttype: TransformType = TransformType.NoTransform #TransformType enum
        self.vXYZn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vActingXYZn = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.mT = torch.eye(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        
        self.vFb = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vM = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        
    def SetTransformType(self, ttype: TransformType):
        self.ttype = ttype

    def SetLocation(self, vv: torch.Tensor):
        self.vXYZn = vv
        self.SetActingLocation(vv)

    def SetActingLocation(self, vv: torch.Tensor):
        self.vActingXYZn = vv

    def SetAnglesToBody(self, angles: torch.Tensor):
        if self.ttype == TransformType.Custom:
            self.vOrient[..., EulerAngles.Phi] = angles[..., EulerAngles.Phi]
            self.vOrient[..., EulerAngles.Tht] = angles[..., EulerAngles.Tht]
            self.vOrient[..., EulerAngles.Psi] = angles[..., EulerAngles.Psi]
            self.UpdateCustomTransformationMatrix()
    
    def UpdateCustomTransformationMatrix(self):
        pitch = self.vOrient[..., 0]
        roll = self.vOrient[..., 1]
        yaw = self.vOrient[..., 2]

        # Calculate trigonometric values
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # Calculate intermediate terms
        srsp = sr * sp
        crcy = cr * cy
        crsy = cr * sy

        # Fill the transformation matrix mT
        self.mT[..., 0, 0] = cp * cy
        self.mT[..., 1, 0] = cp * sy
        self.mT[..., 2, 0] = -sp

        self.mT[..., 0, 1] = srsp * cy - crsy
        self.mT[..., 1, 1] = srsp * sy + crcy
        self.mT[..., 2, 1] = sr * cp

        self.mT[..., 0, 2] = crcy * sp + sr * sy
        self.mT[..., 1, 2] = crsy * sp - sr * cy
        self.mT[..., 2, 2] = cr * cp