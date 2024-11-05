from typing import Optional

import torch

from jsbsim_parallel.models.standard_atmosphere import StandardAtmosphere
from jsbsim_parallel.models.unit_conversions import UnitConversions

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

def RankineToCelsius(rankine: torch.Tensor):
    return (rankine - 491.67)/1.8

class Auxiliary:
    def __init__(self, atmosphere: StandardAtmosphere, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self.batch_size = batch_size
        self._in = AuxiliaryInputs(self.device, self.size)

        # Scalar variables with a single element in an extra dimension
        self.vcas = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.veas = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.pt = atmosphere.StdDaySLpressure
        self.tat = atmosphere.StdDaySLtemperature
        self.tatc = RankineToCelsius(self.tat)
        self.Vt = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Vground = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Mach = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.MachU = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.qbar = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.qbarUW = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.qbarUV = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Re = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alpha = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.beta = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.adot = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.bdot = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.psigt = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.gamma = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Nx = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Ny = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Nz = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.hoverbcg = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.hoverbmac = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        # 3x3 matrices
        self.mTw2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.mTb2w = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

        # 3-element vectors
        self.vPilotAccel = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vPilotAccelN = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vNcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vNwcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vAeroPQR = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vAeroUVW = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vEulerRates = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vMachUVW = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vLocationVRP = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.units = UnitConversions.get_instance()

    def GetTw2b(self):
        return self.mTw2b
    
    def GetVt(self):
        return self.Vt
    
    def Getalpha(self):
        return self.alpha

    def Getbeta(self):
        return self.beta

    def Getqbar(self):
        return self.qbar

    def GetTb2w(self):
        return self.mTb2w
        
    def GetVground(self):
        return self.Vground
    
    def GetVcalibratedKTS(self):
        return self.vcas * self.units.FPS_TO_KTS

    def run(self, holding: bool) -> bool:
        if holding:
            return False
        
        return True

    def init_model(self) -> bool:
        #reset to IC.
        return True

