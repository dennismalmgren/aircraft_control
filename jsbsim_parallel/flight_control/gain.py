from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.real_value import RealValue
from jsbsim_parallel.math.table_1d import Table1D

class Gain(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Gain = None
        self.Table = None
        self.InMin = -1.0
        self.InMax = 1.0
        self.OutMin = 0.0
        self.OutMax = 0.0
        self.ZeroCentered = False

        self.CheckInputNodes(1, 1, element)

        if self.Type == "PURE_GAIN":
            if not element.FindElement("gain"):
                print("No Gain specified (default 1.0)")
        
        gain_element = element.FindElement("gain")
        if gain_element:
            self.Gain = ParameterValue(None, gain_element, device=self.device, batch_size=self.size)
        else:
            self.Gain = RealValue(1.0)

        if self.Type == "AEROSURFACE_SCALE":
            scale_element = element.FindElement("domain")
            if scale_element:
                if scale_element.FindElement("max") and scale_element.FindElement("min"):
                    self.InMax = scale_element.FindElementValueAsNumber("max")
                    self.InMin = scale_element.FindElementValueAsNumber("min")

            scale_element = element.FindElement("range")
            if not scale_element:
                raise Exception("No range supplied for aerosurface scale component")
            if scale_element.FindElement("max") and scale_element.FindElement("min"):
                self.OutMax = scale_element.FindElementValueAsNumber("max")
                self.OutMin = scale_element.FindElementValueAsNumber("min")
            else:
                raise Exception("Max/min output must be supplied for aerosurface scale component")
            
            self.ZeroCentered = True
            zero_centered = element.FindElement("zero_centered")
            if zero_centered:
                sZeroCentered = element.FindElementValue("zero_centered")
                if sZeroCentered == "0" or sZeroCentered == "false":
                    self.ZeroCentered = False

        if self.Type == "SCHEDULED_GAIN":
            if element.FindElement("table"):
                self.Table = Table1D(None, None, element.FindElement("table"), 
                                     device=self.device)
            else:
                raise Exception("A table must be provided for scheduled gain component")
        
        #TODO: BIND
