from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.models.propulsion_models.force import Force, TransformType
from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.input_output.simulator_service import SimulatorService

class ThrusterType(IntEnum):
    Nozzle = 0
    Rotor = 1
    Propeller = 2
    Direct = 3


class ThrusterInputs:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):

        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        

        # Single-value attributes initialized as tensors with an extra dimension for compatibility with batched operations
        self.TotalDeltaT = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.H_agl = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Density = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Pressure = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Soundspeed = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Alpha = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Beta = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Vt = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        
        # Vector attributes (3D) initialized as tensors with shape [*batch_size, 3]
        self.PQRi = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.AeroPQR = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.AeroUVW = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)


class Thruster(Force):
    def __init__(self, mass_balance: MassBalance, 
                 simulator_service: SimulatorService,
                 el: Element, 
                 engine_number: int, *, 
                 device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(mass_balance, simulator_service, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        thruster_element = el.GetParent()
        self.Type = ThrusterType.Direct
        self.SetTransformType(TransformType.Custom)

        self.Name = el.GetAttributeValue("name")
        
        self.GearRatio =  torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.EngineNum = engine_number

        # TODO: Get property manager

        # Determine the initial location and orientation of this thruster and load the
        # thruster with this information.

        element = thruster_element.FindElement("location")
        if element is not None:
            location = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.size)
        else:
            print("Location missing for thruster")
            location = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.SetLocation(location)

        # TODO: Tie properties

        element = thruster_element.FindElement("pointing")

        if element is not None:
           # This defines a fixed nozzle that has no public interface property to gimbal or reverse it.
           # The specification of RAD here is superfluous,
           # and simply precludes a conversion.
            pointing = element.FindElementTripletConvertTo("RAD", device=self.device, batch_size=self.size)
            self.mT[..., 0, 0] = pointing[..., 0]
            self.mT[..., 1, 0] = pointing[..., 1]
            self.mT[..., 2, 0] = pointing[..., 2]
        else:
            element = thruster_element.FindElement("orient")
            if element is not None:
                orientation = element.FindElementTripletConvertTo("RAD", device=self.device, batch_size=self.size)
            else:
                orientation = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
            self.SetAnglesToBody(orientation)

            #TODO: Properties
        
        #TODO: Reset to ic

        #TODO: These seem left at 0.
        self.Thrust = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.PowerRequired = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        
        self.ThrustCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.ReverserAngle = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        
