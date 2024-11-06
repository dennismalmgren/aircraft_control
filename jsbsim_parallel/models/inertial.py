from enum import Enum
from typing import List, Optional

import torch

from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.math.location import Location

class GravType(Enum):
    ''' Gravitational model type
    '''

    # Evaluate gravity using Newton's classical formula assuming the Earth is
    # spherical
    Standard: int = 0
    # Evaluate gravity using WGS84 formulas that take the Earth oblateness
    # into account
    WGS84: int = 1


class InertialInputs:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Use batch_size for initialization if provided, else default to a single instance.
        size = batch_size if batch_size is not None else torch.Size([])
        self.Position = Location(device=device, batch_size=size)

class Inertial:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        units = UnitConversions.get_instance()
        self._rotation_rate = torch.tensor(0.00007292115, dtype=torch.float64, device=device)
        self._GM = torch.tensor(14.0764417572E15, dtype=torch.float64, device=device) # WGS84 value
        self._J2 = torch.tensor(1.08262982E-03, dtype=torch.float64, device=device) # WGS84 value for J2
        self._a = torch.tensor(20925646.32546, dtype=torch.float64, device=device) # WGS84 semimajor axis length in feet
        self._b = torch.tensor(20855486.5951, dtype=torch.float64, device=device) # WGS84 semiminor axis length in feet
        self._gravType = GravType.WGS84
        self.vOmegaPlanet = torch.tensor([0.0, 0.0, self._rotation_rate], dtype=torch.float64, device=device)
        self.vGravAccel = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        
        # Standard gravity (9.80665 m/s^2) in ft/s^2 which is the gravity at 45 deg.
        # of latitude (see ISA 1976 and Steven & Lewis)
        # It includes the centripetal acceleration.        
        self._gAccelReference = 9.80665 * units.FT_TO_M
        self._in = InertialInputs(device=device, batch_size=size)

    def omega_planet(self):
        return self.vOmegaPlanet
    
    def semi_major(self):
        return self._a
    
    def semi_minor(self):
        return self._b

    def GM(self):
        return self._GM

    def standard_gravity(self):
        return self._gAccelReference    
    
    def GetGravity(self) -> torch.Tensor:
        return self.vGravAccel
    
    def run(holding: bool) -> bool:
        pass

    def init_model(self) -> bool:
        #should call fgmodelfunctions::initmodel (base class fn)
        #reset to IC.
        return True
    
    def set_altitude_AGL(self, location: Location, altitude:torch.Tensor):
        pass

    def get_altitude_AGL(self, location: Location):
        pass
        #return GroundCallback->GetAGLevel(location, lDummy, vDummy, vDummy,
        #                              vDummy)
        # pass

