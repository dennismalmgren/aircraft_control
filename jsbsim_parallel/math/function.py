from typing import Optional, List, Callable
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.property_value import PropertyValue
from jsbsim_parallel.math.table_2d import Table2D



class Function(Parameter):
    
    #todo: propertyvalue constructor.
    #currently "val"
    def __init__(self, el: Element, prefix: str = "", var: PropertyValue = None, *, 
                 device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.cached = False
        self.cachedValue = torch.full((*self.size, 1), float("-inf"), dtype=torch.float64, device=self.device)
        self.Parameters: List[Parameter] = []
        self.Name = ""
        #TODO: pCopyTo propertynode, pNode propertynode, propertymanager

        self.Load(el, var, prefix)
        self.CheckMinArguments(el, 1)
        self.CheckMaxArguments(el, 1)

        sCopyTo = el.GetAttributeValue("copyto")
        if sCopyTo and "#" in sCopyTo:
            if prefix.isdigit():
                sCopyTo = sCopyTo.replace("#", prefix)
            else:
                print("Illegal use fo the special character #")
                return
        

        #TODO: Load self.pCopyTo using sCopyTo.

    def GetName(self) -> str:
        return self.Name
    
    def Load(self, el: Element, var: PropertyValue, Prefix: str = ""):
        self.Name = el.GetAttributeValue("name")
        element = el.GetElement()

        def operation_sum(self, element):
            operands = []
            for child in list(element):
                operands.append(self.parse_element(child))
            return torch.stack(operands).sum(dim=0)

        while element:
            operation = element.GetName()

            # data types
            if operation in ["property", "p"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
                # property_name = element.GetDataLine()
                # if var and property_name.strip() == "#":
                #     self.Parameters.append(var)
                # else:
                #     if "#" in property_name:
                #         if Prefix.isdigit():
                #             property_name = property_name.replace("#", Prefix)
                #         else:
                #             print("Illegal use of the special character #")
                #             raise Exception("Illegal use of the special character #")
                
                # if element.HasAttribute("apply"):
                #     #template fnction
                #     raise Exception("Unsupported attribute apply")
            elif operation in ["value", "v"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["pi"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["table", "t"]:
                call_type = element.GetAttributeValue("type")
                if call_type == "internal":
                    print("An internal table cannot be nested within a function.")
                    raise Exception("An internal table cannot be nested within a function.")
                table = Table2D(el=element, prefix=Prefix, device=self.device)
                self.Parameters.append(table)
                #print("Unsupported operation")
                #raise Exception("Unsupported operation " + operation)
            
            elif operation in ["product"]:
                def create_product_fn():
                    def fn(self, parameters: List[Parameter]) -> torch.Tensor:
                        result = torch.ones(self.size, 1, dtype=torch.float64, device=self.device)
                        for parameter in parameters:
                            result *= parameter.GetValue()
                        return result
                    return fn
                self.Parameters.append(OperationParameter(create_product_fn(), self.Parameters))
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["sum"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["avg"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["difference"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["min"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["max"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["and"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["or"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["quotient"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["pow"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["toradians"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["todegrees"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["sqrt"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["log2"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["ln"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["log10"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["sign"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["exp"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["abs"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["sin"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["cos"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["tan"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["asin"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["acos"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["atan"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["floor"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["ceil"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["fmod"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["roundmultiple"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["atan2"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["mod"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["fraction"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["integer"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["lt"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["le"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["gt"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["ge"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["eq"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["nq"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["not"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["ifthen"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["random"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["urandom"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["switch"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["interpolate1d"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["rotation_alpha_local"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["rotation_beta_local"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["rotation_gamma_local"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["rotation_bf_to_wf"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation in ["rotation_wf_to_bf"]:
                print("Unsupported operation")
                raise Exception("Unsupported operation " + operation)
            elif operation not in ["description"]:
                print("Bad operation")
                raise Exception("Bad operation " + operation)
            
            

            element = el.GetNextElement()

    def CheckMinArguments(self, el: Element, _min: int):
        if len(self.Parameters) < _min:
            print(el.GetName() + " should have at least 1 argument")

    def CheckMaxArguments(self, el: Element, _max: int):
        if len(self.Parameters) > _max:
            print(el.GetName() + " should have no more than 1 argument")

    #  el: Element, prefix: str = "", var: PropertyValue = None, *, 
    #              device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):

class OperationParameter(Function):
    def __init__(self, f: Callable, el:Element, prefix: str = "", var: PropertyValue=None, *, 
                  device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        super().__init__(el, prefix, var, device=device, batch_size=batch_size)
        self.f = f

        # TODO: Bind

    def GetValue(self) -> torch.Tensor:
        result = self.cachedValue if self.cached else self.f()
        return result
