from typing import Optional, Dict
from enum import IntEnum

import torch

class AxisType(IntEnum):
    NoAxis = 0
    Wind = 1
    BodyAxialNormal = 2
    BodyXYZ = 3
    Stability = 4


class AerodynamicsInputs:
    def __init__(self, device='cpu', batch_size: Optional[torch.Size] = None):
        """
        Initializes the Inputs structure with placeholders for aerodynamic and body properties.
        
        :param device: Device on which tensors are stored ('cpu' or 'cuda').
        :param batch_size: Optional batch size for batched inputs.
        """
        # Set size as batch size or default to a single dimension
        size = batch_size if batch_size is not None else torch.Size([])

        # Scalars
        self.Alpha = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Beta = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Vt = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Qbar = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingarea = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingspan = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingchord = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.Wingincidence = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        
        # Vectors and Matrices
        self.RPBody = torch.zeros(*size, 3, dtype=torch.float64, device=device)  # 3-element vector
        self.Tb2w = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)  # 3x3 matrix
        self.Tw2b = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)  # 3x3 matrix

class Aerodynamics:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self.forceAxisType = AxisType.NoAxis
        self.momentAxisType = AxisType.NoAxis
        self.AxisIdx: Dict[str, int] = dict()

        self.AxisIdx["DRAG"]   = 0
        self.AxisIdx["SIDE"]   = 1
        self.AxisIdx["LIFT"]   = 2
        self.AxisIdx["ROLL"]   = 3
        self.AxisIdx["PITCH"]  = 4
        self.AxisIdx["YAW"]    = 5

        self.AxisIdx["AXIAL"]  = 0
        self.AxisIdx["NORMAL"] = 2

        self.AxisIdx["X"] = 0
        self.AxisIdx["Y"] = 1
        self.AxisIdx["Z"] = 2
        self.AeroFunctions = [] #6 items long
        self.AeroFunctionsAtCG = [] # 6 items long

        #fgfunction AeroRPShift (?)
        #fgfunction-list AeroFunctions
        # Transformation matrices
        self.Ts2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)  # 3x3 matrix
        self.Tb2s = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)  # 3x3 matrix

        # Force and moment vectors
        self.vFnative = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vFw = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vFnativeAtCG = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vForcesAtCG = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vMomentsMRC = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vMomentsMRCBodyXYZ = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vDXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vDeltaRP = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

        # Aerodynamic parameters
        self.alphaclmax = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphaclmin = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphaclmax0 = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphaclmin0 = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphahystmax = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphahystmin = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.impending_stall = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.stall_hyst = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.bi2vel = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.ci2vel = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.alphaw = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.clsq = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.lod = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.qbar_area = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        #self.AeroRPShift = 0#function pointer. still don't know what.

        self._in = AerodynamicsInputs(device, batch_size)


    def GetvFw(self) -> torch.Tensor:
        return self.vFw
    
    def GetForces(self) -> torch.Tensor:
        return self.vForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vMoments

    def run(self, holding: bool) -> bool:
        pass

    def init_model(self) -> bool:

        self.impending_stall.zero_()
        self.stall_hyst.zero_()
        self.alphaclmin.zero_()
        self.alphaclmin0.zero_()
        self.alphaclmax.zero_()
        self.alphaclmax0.zero_()
        self.alphahystmin.zero_()
        self.alphahystmax.zero_()
        self.clsq.zero_()
        self.lod.zero_()
        self.alphaw.zero_()
        self.bi2vel.zero_()
        self.ci2vel.zero_()
        #self.AeroRPShift = 0#function pointer. still don't know what.
        self.vDeltaRP.zero_()
        self.vForces.zero_()
        self.vMoments.zero_()

        #reset to IC.
        return True