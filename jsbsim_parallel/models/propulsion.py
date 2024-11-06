from typing import Optional, List
from enum import Enum

import torch

from jsbsim_parallel.models.propulsion_models.tank import Tank
from jsbsim_parallel.models.propulsion_models.engine import Engine, EngineInputs
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.models.unit_conversions import UnitConversions

class Propulsion(ModelBase):
    def __init__(self, mass_balance: MassBalance, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.mass_balance = mass_balance
        
        self.active_engine = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device) # -1: ALL, 0: Engine 1, 1: Engine 2 ...
        self.tankJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        self.dump_rate = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.refuel_rate = torch.tensor([6000.0], dtype=torch.float64, device=self.device).expand(*self.size, 1)
        self.fuel_freeze = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.tanks: List[Tank] = []
        self.engines: List[Engine] = []
        self.total_fuel_quantity = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.total_oxidizer_quantity = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.refuel = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.dump = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self._in = EngineInputs(device = self.device, batch_size = self.size)
        self.vXYZtank_arm = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.units = UnitConversions.get_instance()

    def CalculateTankInertias(self) -> torch.Tensor:
        if len(self.tanks) == 0:
            return self.tankJ
        else:
            self.tankJ.zero_()
            for tank in self.tanks:
                self.tankJ += self.mass_balance.GetPointmassInertia(self.units.LB_TO_SLUG * \
                                                                    tank.GetContents(),
                                                                    tank.GetXYZ())
        return self.tankJ
    

    def GetTanksMoment(self) -> torch.Tensor:
        self.vXYZtank_arm.zero_()
        for tank in self.tanks:
            self.vXYZtank_arm += tank.GetXYZ() * tank.GetContents()

    def GetTanksWeight(self) -> torch.Tensor:
        Tw = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        for tank in self.tanks:
            Tw += tank.GetContents()
        return Tw
    
    def GetForces(self) -> torch.Tensor:
        return self.vForces
    
    def GetMoments(self) -> torch.Tensor:
        return self.vMoments
    
    def init_model(self) -> bool:      
        self.vForces = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vMoments = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        for tank in self.tanks:
            tank.reset_to_ic()
        self.total_fuel_quantity = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.total_oxidizer_quantity = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        self.refuel = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.dump = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)

        for engine in self.engines:
            engine.reset_to_ic()

        return True
    
    def run(holding: bool) -> bool:
        if holding:
            return False
        
        return True