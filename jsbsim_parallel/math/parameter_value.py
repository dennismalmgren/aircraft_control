from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element, isfloat
from jsbsim_parallel.math.property_value import PropertyValue
from jsbsim_parallel.math.real_value import RealValue
from jsbsim_parallel.math.parameter import Parameter

class ParameterValue(Parameter):
    def __init__(self,
                 val: str = None,
                 el: Element = None,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        
        if not val:
            val = el.GetDataLine()
            if el.GetNumDataLines() != 1 or len(val) == 0:
                raise Exception("Illegal argument for parameter value")
        if isfloat(val):
            self.param = RealValue(float(val))
        else:
            self.param = PropertyValue(val, el, device=device, batch_size=self.size)

        
        

    def GetValue(self):
        return self.param.GetValue()
    
    def IsConstant(self):
        return self.param.IsConstant()
    
    def GetName(self):
        if self.param is PropertyValue:
            return self.param.GetNameWithSign()
        else:
            return self.param.GetName()