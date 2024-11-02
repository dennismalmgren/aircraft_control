from typing import Optional

import torch

class AuxiliaryInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Use batch_size for initialization if provided, else default to a single instance.
        size = batch_size if batch_size is not None else torch.Size([])

        # Scalars as tensors, supporting batched computation if batch_size is specified
        self.Pressure = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Density = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.StdDaySLsoundspeed = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.SoundSpeed = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.KinematicViscosity = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.DistanceAGL = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingspan = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingchord = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.StandardGravity = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Mass = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.CosTht = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.SinTht = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.CosPhi = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.SinPhi = torch.zeros(*size, 1, dtype=torch.float64, device=device)

        # Matrices
        self.Tl2b = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.Tb2l = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)

        # Vectors
        self.vPQR = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vPQRi = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vPQRidot = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vUVW = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vUVWdot = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vVel = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vBodyAccel = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.ToEyePt = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.RPBody = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.VRPBody = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vFw = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vLocation = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.TotalWindNED = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.TurbPQR = torch.zeros(*size, 3, dtype=torch.float64, device=device)

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        self.Pressure = self.Pressure.to(device)
        self.Density = self.Density.to(device)
        self.Temperature = self.Temperature.to(device)
        self.StdDaySLsoundspeed = self.StdDaySLsoundspeed.to(device)
        self.SoundSpeed = self.SoundSpeed.to(device)
        self.KinematicViscosity = self.KinematicViscosity.to(device)
        self.DistanceAGL = self.DistanceAGL.to(device)
        self.Wingspan = self.Wingspan.to(device)
        self.Wingchord = self.Wingchord.to(device)
        self.StandardGravity = self.StandardGravity.to(device)
        self.Mass = self.Mass.to(device)
        self.CosTht = self.CosTht.to(device)
        self.SinTht = self.SinTht.to(device)
        self.CosPhi = self.CosPhi.to(device)
        self.SinPhi = self.SinPhi.to(device)

        self.Tl2b = self.Tl2b.to(device)
        self.Tb2l = self.Tb2l.to(device)
        self.vPQR = self.vPQR.to(device)
        self.vPQRi = self.vPQRi.to(device)
        self.vPQRidot = self.vPQRidot.to(device)
        self.vUVW = self.vUVW.to(device)
        self.vUVWdot = self.vUVWdot.to(device)
        self.vVel = self.vVel.to(device)
        self.vBodyAccel = self.vBodyAccel.to(device)
        self.ToEyePt = self.ToEyePt.to(device)
        self.RPBody = self.RPBody.to(device)
        self.VRPBody = self.VRPBody.to(device)
        self.vFw = self.vFw.to(device)
        self.vLocation = self.vLocation.to(device)
        self.TotalWindNED = self.TotalWindNED.to(device)
        self.TurbPQR = self.TurbPQR.to(device)

class Auxiliary:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self.batch_size = batch_size
        self._in = AuxiliaryInputs(self.device, self.size)
        self.Tw2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        self.Vt = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

    def get_Tw2b(self):
        return self.Tw2b
    
    def get_Vt(self):
        return self.Vt
    
    def run(holding: bool) -> bool:
        pass

    def init_model() -> bool:
        #reset to IC.
        return True

