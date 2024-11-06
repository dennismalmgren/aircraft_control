from typing import Optional
import torch

from jsbsim_parallel.models.propagate import Propagate 
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class MassBalanceInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Define the size as a batch_size if provided, otherwise scalar
        size = batch_size if batch_size is not None else torch.Size([])

        self.GasMass = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TanksWeight = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.GasMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GasInertia = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.TanksMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.TankInertia = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.WOW = torch.zeros(*size, 1, dtype=torch.bool, device=device)

class MassBalance(ModelBase):
    def __init__(self, propagate: Propagate, 
                                  path_provider: ModelPathProvider,
    *, device, batch_size: Optional[torch.Size] = None):
        super().__init__(path_provider, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.propagate = propagate
        self.weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.empty_weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.mass = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        
        self.vbaseXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.baseJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.mJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.mJinv = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.pmJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.units = UnitConversions.get_instance()
        self._in = MassBalanceInputs(device=device, batch_size=batch_size)

    def GetJinv(self) -> torch.Tensor:
        return self.mJinv
    
    def GetJ(self) -> torch.Tensor:
        return self.mJ
    
    # Conversion from the structural frame to the body frame.
    def StructuralToBody(self, r: torch.Tensor) -> torch.Tensor:
        '''
         Under the assumption that in the structural frame the:
        
         - X-axis is directed afterwards,
         - Y-axis is directed towards the right,
         - Z-axis is directed upwards,
        
         (as documented in http://jsbsim.sourceforge.net/JSBSimCoordinates.pdf)
         we have to subtract first the center of gravity of the plane which
         is also defined in the structural frame:
        
           FGColumnVector3 cgOff = r - vXYZcg;
        
         Next, we do a change of units:
        
           cgOff *= inchtoft;
        
         And then a 180 degree rotation is done about the Y axis so that the:
        
         - X-axis is directed forward,
         - Y-axis is directed towards the right,
         - Z-axis is directed downward.
        
         This is needed because the structural and body frames are 180 degrees apart.
        '''
        return torch.cat([
            (r[..., 0] - self.vXYZcg[..., 0]).unsqueeze(-1),
            (-r[..., 1] - self.vXYZcg[..., 1]).unsqueeze(-1),
            (-r[..., 2] - self.vXYZcg[..., 2]).unsqueeze(-1)
        ], dim=-1)* self.units.INCH_TO_FT
    
    def GetPointmassInertia(self, mass_sl: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        v = self.StructuralToBody(r)
        sv = mass_sl * v
        xx = sv[..., 0] * v[..., 0]
        yy = sv[..., 1] * v[..., 1]
        zz = sv[..., 2] * v[..., 2]
        xy = -sv[..., 0] * v[..., 1]
        xz = -sv[..., 0] * v[..., 2]
        yz = -sv[..., 1] * v[..., 2]
        result = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        result[..., 0, 0] = yy + zz
        result[..., 0, 1] = xy
        result[..., 0, 2] = xz
        result[..., 1, 0] = xy
        result[..., 1, 1] = xx + zz
        result[..., 1, 2] = yz
        result[..., 2, 0] = xz
        result[..., 2, 1] = yz
        result[..., 2, 2] = xx + yy
        return result
    
    def GetMass(self) -> torch.Tensor:
        return self.mass

    def GetXYZcg(self):
        return self.vXYZcg
    
    def GetEmptyWeight(self):
        return self.empty_weight
    
    def run(holding: bool) -> bool:
        if (holding):
            return True
        return False

    def init_model(self) -> bool:
        if not super().init_model():
            return False
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        
        return True