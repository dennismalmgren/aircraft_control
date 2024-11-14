from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.input_output.element import Element

class PropertyValue(Parameter):
    def __init__(self, propertyName: str, el: Element, *, 
                 device: torch.device = torch.device("cpu"), 
                 batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Name = propertyName
        if propertyName[0] == "-":
            propertyName = propertyName[1:]
            self.Sign = -1.0
        else:
            self.Sign = 1.0
        
        #TODO: check if property exists in property manager
        
    def GetNameWithSign(self):
        if self.Sign < 0.0:
            return "-" + self.Name
        else:
            return self.Name
        
    def GetName(self):
        # TODO: CHECK IF PROPERTY NODE
        return self.Name
