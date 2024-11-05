from typing import Optional

import torch

class ModelBase:
    def __init__(self, *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        size = batch_size if batch_size is not None else torch.Size([])
        self.rate = torch.ones(*size, 1, dtype=torch.float64, device=device) #todo: probably a scalar
    
    def GetRate(self):
        return self.rate
    
    def init_model(self):
        pass
