from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element, isfloat
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.condition import Condition

class SwitchTest:
    def __init__(self):
        self.condition: Condition = None
        self.Default: bool = False
        self.OutputValue: Parameter = None
    
    def setTestValue(self, val: str, Name: str, el: Element, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        if len(val) == 0:
            raise Exception("No Value supplied for switch component")
        else:
            self.OutputValue = ParameterValue(val, el, device=device, batch_size=batch_size)

    def GetOutputName(self) -> str:
        return self.OutputValue.GetName()
    
class Switch(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.tests: List[SwitchTest] = []
        self.initialized = False

        # TODO: Bind
        test_element = element.FindElement("default")
        if test_element:
            current_test = SwitchTest()
            val = test_element.GetAttributeValue("value")
            current_test.setTestValue(val, self.Name, test_element, device=self.device, batch_size=self.size)
            current_test.Default = True
            if self.delay > 0 and isfloat(val.strip()): 
                #if there is a delay, initialize the delay buffer to the default value
                #for the switch if that value is a number
                v = float(val)
                for i in range(self.delay-1):
                    self.output_array[i] = torch.full((*self.size, 1), v, dtype=torch.float64, device=self.device)
            self.tests.append(current_test)
        
        test_element = element.FindElement("test")
        while test_element:
            current_test = SwitchTest()
            condition = Condition(test_element, None, device=self.device, batch_size=self.size)
            current_test.condition = condition

            val = test_element.GetAttributeValue("value")
            current_test.setTestValue(val, self.Name, test_element, device=self.device, batch_size=self.size)
            self.tests.append(current_test)
            test_element = test_element.GetNextElement()