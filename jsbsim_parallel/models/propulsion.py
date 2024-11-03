from typing import Optional, List
from enum import Enum

import torch

from jsbsim_parallel.models.propulsion_models.tank import Tank
from jsbsim_parallel.models.propulsion_models.engine import Engine, EngineInputs


class Propulsion:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
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
        
        pass