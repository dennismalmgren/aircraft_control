from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.real_value import RealValue
from jsbsim_parallel.math.table_1d import Table1D
from jsbsim_parallel.math.function import Function

class FCSFunction(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        function_element = element.FindElement("function")
        if function_element:
            self.function = Function(function_element, device=self.device, batch_size=self.size)
        else:
            raise Exception("FCS function should contain a function element")
        

        #TODO: Bind
