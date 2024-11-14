from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.math.property_value import PropertyValue
from jsbsim_parallel.math.table_2d import Table2D

class Comparison(IntEnum):
    Undef = 0
    EQ = 1
    NE = 2
    GT = 3
    GE = 4
    LT = 5
    LE = 6

class Logic(IntEnum):
    Undef = 0
    AND = 1
    OR = 2

class Condition:
    def __init__(self,
                 element: Element = None,     
                 data: str = None,         
                 *,
                 device: torch.device,
                 batch_size: Optional[torch.Size] = None):
        
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Logic = Logic.Undef
        self.Comparison = Comparison.Undef
        self.TestParam1 = None
        self.TestParam2 = None
        self.Conditions: List[Condition] = []
        if data is None:
            logic = element.GetAttributeValue("logic")
            if len(logic) > 0:
                if logic == "OR":
                    self.Logic = Logic.OR
                elif logic == "AND":
                    self.Logic = Logic.AND
                else:
                    raise Exception("Illegal logic")
            else:
                self.Logic = Logic.AND

            if self.Logic == Logic.Undef:
                raise Exception("Illegal logic")
        
            for i in range(element.GetNumDataLines()):
                data = element.GetDataLine(i)
                condition = Condition(None, data, device=device, batch_size=self.size)
                self.Conditions.append(condition)

            condition_element = element.GetElement()
            elName = element.GetName()
            while condition_element:
                tagName = condition_element.GetName()

                if tagName != elName:
                    raise Exception("Unrecognized tag")
                condition = Condition(condition_element, None, device=device, batch_size=self.size)
                self.Conditions.append(condition)
                condition_element = condition_element.GetNextElement()
            
            if len(self.Conditions) == 0:
                raise Exception("Empty conditional")
        else:
            mComparison = {
                "!=": Comparison.NE,
                "<":  Comparison.LT,
                "<=": Comparison.LE,
                "==": Comparison.EQ,
                ">":  Comparison.GT,
                ">=": Comparison.GE,
                "EQ": Comparison.EQ,
                "GE": Comparison.GE,
                "GT": Comparison.GT,
                "LE": Comparison.LE,
                "LT": Comparison.LT,
                "NE": Comparison.NE,
                "eq": Comparison.EQ,
                "ge": Comparison.GE,
                "gt": Comparison.GT,
                "le": Comparison.LE,
                "lt": Comparison.LT,
                "ne": Comparison.NE,
            }

            test_strings = data.split()
            if len(test_strings) == 3:
                self.TestParam1 = PropertyValue(test_strings[0], element, device=device, batch_size=self.size)      
                conditional = test_strings[1]
                self.TestParam2 = ParameterValue(test_strings[2], element, device=device, batch_size=self.size)
                if conditional not in mComparison:
                    raise Exception("Illegal conditional")
                self.Comparison = mComparison[conditional]
            else:
                raise Exception("Illegal condition, incorrect number of test elements")
            