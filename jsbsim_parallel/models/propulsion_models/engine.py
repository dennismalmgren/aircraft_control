from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.models.propulsion_models.thruster import Thruster
from jsbsim_parallel.input_output.simulator_service import SimulatorService
from jsbsim_parallel.models.mass_balance import MassBalance

class EngineType(IntEnum):
    Unknown = 1
    Rocket = 2
    Piston = 3
    Turbine = 4
    Turboprop = 5
    Electric = 6
    
class EngineInputs:
    def __init__(self, *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        # Define the size for batch support
        size = batch_size if batch_size is not None else torch.Size([])

        # Scalar properties
        self.Pressure = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.PressureRatio = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Density = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.DensityRatio = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Soundspeed = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TotalPressure = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TAT_c = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Vt = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Vc = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.qbar = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.alpha = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.beta = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.H_agl = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TotalDeltaT = torch.zeros(*size, 1, dtype=torch.float64, device=device)

        # Vector properties (FGColumnVector3 equivalent)
        self.AeroUVW = torch.zeros(*size, 3, 1, dtype=torch.float64, device=device)
        self.AeroPQR = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PQRi = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # List properties for variable-length data
        self.ThrottleCmd = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.MixtureCmd = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.ThrottlePos = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.MixturePos = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]
        self.PropAdvance = [torch.zeros(*size, 1, dtype=torch.float64, device=device)]

        self.PropFeather = [torch.zeros(*size, 1, dtype=torch.bool, device=device)]  
        
class Engine(ModelBase):
    def __init__(self, 
                 mass_balance: MassBalance,
                 simulator_service: SimulatorService,
                 engine_number: int, 
                 inputs: EngineInputs, *, 
                 device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(simulator_service, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.mass_balance = mass_balance
        self.simulator_service = simulator_service
        self.EngineNumber = engine_number
        self._in = inputs
        self.Type = EngineType.Unknown
        self.SLFuelFlowMax = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.FuelExpended = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.MaxThrottle = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        self.MinThrottle = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.FuelDensity = torch.full((*self.size, 1), 6.02, dtype=torch.float64, device=device)
        self.SourceTanks: List[int] = []

    def Load(self, engine_element:Element):
        parent_element = engine_element.GetParent()
        
        # Get Property Manager
        self.Name = engine_element.GetName()

        #call model functions loader.
        #super().Load(...)
        super().PreLoad(engine_element, str(self.EngineNumber))
        
        # If engine location and/or orientation is supplied issue a warning since they
        # are ignored. What counts is the location and orientation of the thruster.
        local_element = parent_element.FindElement("location")
        if local_element is not None:
            print("WARNING: Engine location is ignored, only thruster location is used.")
        
        local_element = parent_element.FindElement("orient")
        if local_element is not None:
            print("WARNING: Engine orientation is ignored, only thruster orientation is used.")

        # Load thruster
        local_element = parent_element.FindElement("thruster")
        if local_element is not None:
            self.LoadThruster(local_element)
        else:
            print("WARNING: No thruster found")

        # TODO: ResetToIC
        
        # Load feed tank[s] references
        local_element = parent_element.FindElement("feed")
        while local_element is not None:
            tankId = int(local_element.GetDataAsNumber())
            self.SourceTanks.append(tankId)
            local_element = parent_element.FindNextElement("feed")

        # TODO: Tie properties

        super().PostLoad(engine_element, str(self.EngineNumber))


    def LoadThruster(self, thruster_element: Element):
        if thruster_element.FindElement("direct"):
            document = thruster_element.FindElement("direct")
            self.Thruster = Thruster(self.mass_balance, 
                                     self.simulator_service,
                                     document, 
                                     self.EngineNumber, 
                                     device=self.device, batch_size=self.size)
        else:
            print("Unsupported thruster type")
            raise Exception("Unsupported thruster type")

    def SetFuelDensity(self, fuelDensity: torch.Tensor):
        self.FuelDensity = fuelDensity

    def reset_to_ic(self):
        pass