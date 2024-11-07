from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class AircraftInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Use batch_size for initialization if provided, else default to a single instance.
        size = batch_size if batch_size is not None else torch.Size([])
        self.AeroForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PropForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GroundForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.ExternalForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.BuoyantForce = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.AeroMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.PropMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GroundMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.ExternalMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.BuoyantMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)

class Aircraft(ModelBase):
    def __init__(self, 
                 path_provider: ModelPathProvider,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size if batch_size is not None else torch.Size([])
        super().__init__(path_provider, device=device, batch_size=batch_size)

        self.path_provider = path_provider
        self._in = AircraftInputs(device, batch_size)

        self.vMoments = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vForces = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZrp = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZvrp = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vXYZep = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)
        self.vDXYZcg = torch.zeros(*self.batch_size, 3, dtype=torch.float64, device=device)

        self.WingArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.WingSpan = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.cbar = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.WingIncidence = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.HTailArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.VTailArea = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.HTailArm = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.VTailArm = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.lbarh = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.lbarv = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.vbarh = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        self.vbarv = torch.zeros(*self.batch_size, 1, dtype=torch.float64, device=device)
        # No support for batched aircraft Names yet
        self.aircraft_name = "FGAircraft"

    def SetAircraftName(self, name: str):
        # No support for batched aircraft Names yet
        self.aircraft_name = name
    
    def Load(self, el: Element) -> bool:
        if not super().Upload(el, True):
            return False

        if el.FindElement("wingarea"):
            self.WingArea.fill_(el.FindElementValueAsNumberConvertTo("wingarea", "FT2"))
        if el.FindElement("wingspan"):
            self.WingSpan.fill_(el.FindElementValueAsNumberConvertTo("wingspan", "FT"))
        if el.FindElement("chord"):
            self.cbar.fill_(el.FindElementValueAsNumberConvertTo("chord", "FT"))
        if el.FindElement("wing_incidence"):
            self.WingIncidence.fill_(el.FindElementValueAsNumberConvertTo("wing_incidence", "RAD"))
        if el.FindElement("htailarea"):
            self.HTailArea.fill_(el.FindElementValueAsNumberConvertTo("htailarea", "FT2"))
        if el.FindElement("htailarm"):
            self.HTailArm.fill_(el.FindElementValueAsNumberConvertTo("htailarm", "FT"))
        if el.FindElement("vtailarea"):
            self.VTailArea.fill_(el.FindElementValueAsNumberConvertTo("vtailarea", "FT2"))
        if el.FindElement("vtailarm"):
            self.VTailArm.fill_(el.FindElementValueAsNumberConvertTo("vtailarm", "FT"))

        # Find all LOCATION elements that descend from this METRICS branch of the
        # config file. This would be CG location, eyepoint, etc.
        element = el.FindElement("location")
        while element is not None:
            element_name = element.GetAttributeValue("name")
            if element_name == "AERORP":
                self.vXYZrp = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.batch_size)
            elif element_name == "EYEPOINT":
                self.vXYZep = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.batch_size)
            elif element_name == "VRP":
                self.vXYZrp = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.batch_size)
            
            element = el.FindNextElement("location")

        nonzero_cbar = self.cbar != 0.0
        self.lbarh.zero_()
        self.lbarv.zero_()
        self.lbarh[nonzero_cbar] = self.HTailArm[nonzero_cbar] / self.cbar[nonzero_cbar]
        self.lbarv[nonzero_cbar] = self.VTailArm[nonzero_cbar] / self.cbar[nonzero_cbar]
        nonzero_cbar_and_area = (self.cbar != 0.0) & (self.WingArea != 0.0)
        self.vbarh.zero_()
        self.vbarv.zero_()
        self.vbarh[nonzero_cbar_and_area] = (self.HTailArm[nonzero_cbar_and_area] * self.HTailArea[nonzero_cbar_and_area]) / \
                                             (self.cbar[nonzero_cbar_and_area] * self.WingArea[nonzero_cbar_and_area])
        self.vbarv[nonzero_cbar_and_area] = (self.VTailArm[nonzero_cbar_and_area] * self.VTailArea[nonzero_cbar_and_area]) / \
                                            (self.cbar[nonzero_cbar_and_area] * self.WingArea[nonzero_cbar_and_area])
        
        #and now, postload
        
        super().PostLoad(el)
        return True
    
    def GetForces(self) -> torch.Tensor:
        return self.vForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vMoments
    
    def GetXYZep(self) -> torch.Tensor:
        return self.vXYZep
    
    def GetXYZvrp(self) -> torch.Tensor:
        return self.vXYZvrp
    
    def GetXYZrp(self) -> torch.Tensor:
        return self.vXYZrp
    
    def run(self, holding:bool) -> bool:
        if holding:
            return False
        
        return True

    def init_model(self) -> bool:
        return True