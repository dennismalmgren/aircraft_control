from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.gas_cell import GasCell, GasCellInputs

class BuoyantForces:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Cells: List[GasCell] = []
        self.vTotalForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vTotalMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.gasCellJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.vGasCellXYZ = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vXYZgasCell_arm = torch.zeros(*self.size, 3, dtype=torch.float64, device=device) # [lbs in]
        self._in = GasCellInputs(device, batch_size)
        self.NoneDefined = torch.ones(*self.size, 1, dtype=torch.bool, device=device)

    def GetGasMassInertia(self) -> torch.Tensor:
        if len(self.Cells) == 0:
            return self.gasCellJ
        else:
            self.gasCellJ.zero_()
            for cell in self.Cells:
                self.gasCellJ += cell.GetInertia()
        return self.gasCellJ
    
    def GetGasMassMoment(self) -> torch.Tensor:
        self.vXYZgasCell_arm.zero_()
        for cell in self.Cells:
            self.vXYZgasCell_arm += cell.GetMassMoment()
        return self.vXYZgasCell_arm
    
    def GetGasMass(self) -> torch.Tensor:
        #TODO: Pre-allocate
        result = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        for cell in self.Cells:
            result += cell.GetMass()
        return result

    def GetForces(self) -> torch.Tensor:
        return self.vTotalForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vTotalMoments
    
    def run(self, holding:bool) -> bool:
        if holding:
            return False
        
        return True

    def init_model(self) -> bool:
        return True
    
#         std::vector <FGGasCell*> Cells;
#   // Buoyant forces and moments. Excluding the gas weight.
#   FGColumnVector3 vTotalForces;  // [lbs]
#   FGColumnVector3 vTotalMoments; // [lbs ft]

#   // Gas mass related masses, inertias and moments.
#   FGMatrix33 gasCellJ;             // [slug ft^2]
#   FGColumnVector3 vGasCellXYZ;
#   FGColumnVector3 vXYZgasCell_arm; // [lbs in]

#   bool NoneDefined;
