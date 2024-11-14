from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.real_value import RealValue

class Kinemat(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.CheckInputNodes(1, 1, element)
        self.Output = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.DoScale: bool = True

        self.Detents: List[float] = []
        self.TransitionTimes: List[float] = []

        if element.FindElement("noscale"):
            self.DoScale = False
        
        traverse_element = element.FindElement("traverse")
        setting_element = traverse_element.FindElement("setting")
        while setting_element:
            tmpDetent = setting_element.FindElementValueAsNumber("position")
            tmpTime = setting_element.FindElementValueAsNumber("time")
            self.Detents.append(tmpDetent)
            self.TransitionTimes.append(tmpTime)
            setting_element = traverse_element.FindNextElement("setting")
        
        if len(self.Detents) <= 1:
            raise Exception("Kinematic component must have more than 1 setting element")
        
        # TODO: BIND

    def GetOutputPct(self) -> torch.Tensor:
        return (self.Output - self.Detents[0]) / (self.Detents[-1] - self.Detents[0])
