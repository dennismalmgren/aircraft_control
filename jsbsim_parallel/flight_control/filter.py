from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.flight_control.fcs_component import FCSComponent
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue

class FilterType(IntEnum):
    Lag = 0
    LeadLag = 1
    Order2 = 2
    Washout = 3
    Unknown = 4


class Filter(FCSComponent):
    def __init__(self,
                 fcs: 'FCS',
                 element: Element,
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        super().__init__(fcs, element, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.DynamicFilter = False
        self.Initialize = True

        self.C: List[Parameter] = [None] * 7 #first is not used...

        self.CheckInputNodes(1, 1, element)

        for i in range(1, 7):
            self.ReadFilterCoefficients(element, i)

        if self.Type == "LAG_FILTER":
            self.FilterType = FilterType.Lag
        elif self.Type == "LEAD_LAG_FILTER":
            self.FilterType = FilterType.LeadLag
        elif self.Type == "SECOND_ORDER_FILTER":
            self.FilterType = FilterType.Order2
        elif self.Type == "WASHOUT_FILTER":
            self.FilterType = FilterType.Washout
        else:
            self.FilterType = FilterType.Unknown

        self.CalculateDynamicFilters()

        #TODO: BIND


    def ReadFilterCoefficients(self, element: Element, index: int):
        coefficient = f"c{index}"
        if element.FindElement(coefficient):
            self.C[index] = ParameterValue(None, element.FindElement(coefficient), device=self.device)
        self.DynamicFilter = self.DynamicFilter or not self.C[index].IsConstant()

    def CalculateDynamicFilters(self):
        denom = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        if self.FilterType == FilterType.Lag:
            denom = 2.0 + self.dt * self.C[1].GetValue()
            self.ca = self.dt * self.C[1].GetValue() / denom
            self.cb = (2.0 - self.dt * self.C[1].GetValue()) / denom
        elif self.FilterType == FilterType.LeadLag:
            denom = 2.0 + self.C[3].GetValue() + self.dt * self.C[4].GetValue()
            self.ca = (2.0 * self.C[1].GetValue() + self.dt * self.C[2].GetValue()) / denom
            self.cb = (self.dt * self.C[2].GetValue() - 2.0 * self.C[1].GetValue()) / denom
            self.cc = (2.0 * self.C[3].GetValue() - self.dt * self.C[4].GetValue()) / denom
        elif self.FilterType == FilterType.Order2:
            denom = 4.0 * self.C[4].GetValue() + 2.0 * self.C[5].GetValue() * self.dt + self.C[6].GetValue() * self.dt**2
            self.ca = (4.0 * self.C[1].GetValue() + 2.0 * self.C[2].GetValue() * self.dt + self.C[3].GetValue() * self.dt**2) / denom
            self.cb = (2.0 * self.C[3].GetValue() * self.dt**2 - 8.0 * self.C[1].GetValue()) / denom
            self.cc = (4.0 * self.C[1].GetValue() - 2.0 * self.C[2].GetValue() * self.dt + self.C[3].GetValue() * self.dt**2) / denom
            self.cd = (2.0 * self.C[6].GetValue() * self.dt**2 - 8.0 * self.C[4].GetValue()) / denom
            self.ce = (4.0 * self.C[4].GetValue() - 2.0 * self.C[5].GetValue() * self.dt + self.C[6].GetValue() * self.dt**2) / denom
        elif self.FilterType == FilterType.Washout:
            denom = 2.0 + self.dt * self.C[1].GetValue()
            self.ca = 2.0 / denom
            self.cb = (2.0 - self.dt * self.C[1].GetValue()) / denom
        elif self.FilterType == FilterType.Unknown:
            raise Exception("Unknown filter type")
