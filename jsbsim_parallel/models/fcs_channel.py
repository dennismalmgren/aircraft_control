from typing import Optional, List, Union
from enum import IntEnum

import torch

#from jsbsim_parallel.models.fcs import FCS
from jsbsim_parallel.input_output.property_manager import PropertyNode
from jsbsim_parallel.flight_control.fcs_component import FCSComponent

class FCSChannel:
    def __init__(self,
                 fcs,
                 name: str,
                 execRate: int,
                 propertyNode: PropertyNode = None,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.fcs = fcs
        self.OnOffNode = propertyNode
        self.Name = name
        self.ExecRate = execRate if execRate >= 1 else 1
        # set ExecFrameCountSinceLastRun so that each components are initialized
        self.ExecFrameCountSinceLastRun = self.ExecRate
        self.FCSComponents: List[FCSComponent] = []

    def Add(self, comp: FCSComponent):
        self.FCSComponents.append(comp)

    def GetName(self):
        return self.Name
    