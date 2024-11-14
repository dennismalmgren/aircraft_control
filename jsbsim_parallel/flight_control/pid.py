from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.real_value import RealValue
from jsbsim_parallel.math.property_value import PropertyValue

class IntegrateType(IntEnum):
    NoIntegrate = 0
    RectEuler = 1
    Trapezoidal = 2
    AdamsBashforth2 = 3
    AdamsBashforth3 = 4

class PID(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.I_out_total = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Input_prev = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.Input_prev2 = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        self.IsStandard = False

        self.IntType = IntegrateType.NoIntegrate
        self.ProcessVariableDot: Parameter = None   
        self.Trigger: Parameter = None

        self.CheckInputNodes(1, 1, element)
        pid_type = element.GetAttributeValue("type")
        if pid_type == "standard":
            self.IsStandard = True
        
        el = element.FindElement("kp")
        if el:
            self.Kp = ParameterValue(None, el, device=self.device, batch_size=self.size)
        else:
            self.Kp = RealValue(0.0)

        el = element.FindElement("ki")
        if el:
            integ_type = el.GetAttributeValue("type")
            if integ_type == "rect":
                self.IntType = IntegrateType.RectEuler
            elif integ_type == "trap":
                self.IntType = IntegrateType.Trapezoidal
            elif integ_type == "ab2":
                self.IntType =  IntegrateType.AdamsBashforth2
            elif integ_type == "ab3":
                self.IntType =  IntegrateType.AdamsBashforth3
            else:
                self.IntType = IntegrateType.AdamsBashforth2 # Use default Adams Bashforth 2nd order integration
            self.Ki = ParameterValue(None, el, device=self.device, batch_size=self.size)
        else:
            self.Ki = RealValue(0.0)

        el = element.FindElement("kd")
        if el:
            self.Kd = ParameterValue(None, el, device=self.device, batch_size=self.size)
        else:
            self.Kd = RealValue(0.0)

        el = element.FindElement("pvdot")
        if el:
            self.ProcessVariableDot = PropertyValue(el.GetDataLine(), el, device=self.device, batch_size=self.size)

        el = element.FindElement("trigger")
        if el:
            self.Trigger = PropertyValue(el.GetDataLine(), el, device=self.device, batch_size=self.size)

        #TODO: BIND

    def SetInitialOutput(self, output: torch.Tensor):
        self.I_out_total.copy_(output)
        self.Output.copy_(output)

