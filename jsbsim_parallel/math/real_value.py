from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.math.parameter import Parameter

class RealValue(Parameter):
    def __init__(self, value: float):
        self.Value = value

    def GetName(self) -> str:
        return "constant value " + str(self.Value)
    
    def IsConstant(self) -> bool:
        return True
    
    def GetValue(self) -> torch.Tensor:
        return self.Value