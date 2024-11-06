from typing import Optional, Dict, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.gas_cell import GasCell

class BuoyantForces:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.vTotalForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vTotalMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.Cells: List[GasCell] = []
        self.gasCellJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

#         std::vector <FGGasCell*> Cells;
#   // Buoyant forces and moments. Excluding the gas weight.
#   FGColumnVector3 vTotalForces;  // [lbs]
#   FGColumnVector3 vTotalMoments; // [lbs ft]

#   // Gas mass related masses, inertias and moments.
#   FGMatrix33 gasCellJ;             // [slug ft^2]
#   FGColumnVector3 vGasCellXYZ;
#   FGColumnVector3 vXYZgasCell_arm; // [lbs in]

#   bool NoneDefined;
