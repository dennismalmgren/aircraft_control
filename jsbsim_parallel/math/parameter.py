import torch

class Parameter:
    
    def GetValue(self) -> torch.Tensor:
        pass

    def GetName(self) -> str:
        pass

    def IsConstant(self) -> bool:
        return False
    
    def getDoubleValue(self):
        return self.GetValue()
    
    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self.get_value() * other
        elif isinstance(other, Parameter):
            return self.GetValue() * other.GetValue()
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return other * self.GetValue()
        else:
            return NotImplemented