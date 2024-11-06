from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.external_force import ExternalForce
from jsbsim_parallel.models.propulsion_models.force import Force
from jsbsim_parallel.models import MassBalance

class TemperaturePressureProvider:
    def GetTemperature(self):
        pass

    def GetPressure(self):
        pass

class Element: #TODO
    def __init__(self):
        pass

class GasCellInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Pressure = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.Density = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)
        self.gravity = torch.zeros(*batch_size, 1, dtype=torch.float64, device=device)

class GasType(IntEnum):
    UnknownGas = 0
    Hydrogen = 1
    Helium = 2
    Air = 3

class Ballonet:
    def __init__(self,
                 num: int,
                 parent: TemperaturePressureProvider, 
                 inputs: GasCellInputs,
                 *,
                 device: torch.device, 
                 batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.CellNum = num
        self.Parent = parent
        self.inputs = inputs

        # Structural constants
        self.MaxVolume = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #[ft^3]
        self.MaxOverPressure = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #[lbs/ft^2]
        self.vXYZ = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device) # [in]
        self.Xradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Yradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Zradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Xwidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Ywidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Zwidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.ValveCoefficient = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^4 sec / slug]
        self.HeatTransferCoeff = [] #List of FgFunction
        self.Blowerinput = None

        # Variables
        self.Pressure = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [lbs/ft^2]
        self.Contents = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [mol]
        self.Volume = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^3]
        self.dVolumeIdeal = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^3]
        self.dU = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [lbs ft / sec]
        self.Temperature = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [Rankine]
        self.ValveOpen = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [Rankine]
        self.ballonetJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device) # [slug foot^2]

        # Constants. 
        self.R = torch.tensor(3.4071, dtype=torch.float64, device=self.device) #[lbs ft/(mol Rankine)]
        self.M_air = torch.tensor(0.0019186, dtype=torch.float64, device=self.device) # [slug/mol]
        self.Cv_air = torch.tensor(5.0/2.0, dtype=torch.float64, device=self.device) # [-]

        # TODO: Element extraction


class GasCell(Force, TemperaturePressureProvider):
    def __init__(self, 
                 inputs: GasCellInputs, 
                 element: Element,
                 num: int,
                 mass_balance: MassBalance,
                 *
                 device: torch.device, 
                 batch_size: Optional[torch.Size] = None):
        super().__init__(self, mass_balance=mass_balance, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Type: GasType = GasType.UnknownGas
        self.CellNum = num
        self.MassBalance = mass_balance

        # Structural constants
        self.MaxVolume = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #[ft^3]
        self.MaxOverPressure = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #[lbs/ft^2]
        self.vXYZ = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device) # [in]
        self.Xradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Yradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Zradius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Xwidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Ywidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.Zwidth = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft]
        self.ValveCoefficient = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^4 sec / slug]
        self.HeatTransferCoeff = [] #List of FgFunction
        self.Ballonet = [] #List of Ballonet

        # Variables
        self.Pressure = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [lbs/ft^2]
        self.Contents = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [mol]
        self.Volume = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^3]
        self.dVolumeIdeal = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [ft^3]
        self.Temperature = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [Rankine]
        self.Buoyancy = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [lbs] Note: Gross lift.
                                                                                            # Does not include the weight of the gas itself.
        self.ValveOpen = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # 0 <= ValveOpen <= 1 (or higher).
        self.Mass = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) # [slug]
        self.gasCellJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device) # [slug foot^2]
        self.gasCellM = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device) # [lbs in]
        

        # Constants. 
        self.R = torch.tensor(3.4071, dtype=torch.float64, device=self.device) #[lbs ft/(mol Rankine)]
        self.M_air = torch.tensor(0.0019186, dtype=torch.float64, device=self.device) # [slug/mol]
        self.M_hydrogen = torch.tensor(0.00013841, dtype=torch.float64, device=self.device) # [slug/mol]
        self.M_helium = torch.tensor(0.00027409, dtype=torch.float64, device=self.device) # [slug/mol]


        self.SetTransformType(torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device))

    def GetMassMoment(self) -> torch.Tensor:
        return self.gasCellM    

    def GetMass(self) -> torch.Tensor:
        return self.Mass
    
    def GetInertia(self) -> torch.Tensor:
        return self.gasCellJ
    
    def GetTemperature(self) -> torch.Tensor:
        return self.Temperature
    
    def GetPressure(self) -> torch.Tensor:
        return self.Pressure