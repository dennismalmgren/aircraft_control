from typing import Optional, List
from enum import Enum
import os

import torch

from jsbsim_parallel.models.propulsion_models.tank import Tank, TankType
from jsbsim_parallel.models.propulsion_models.engine import Engine, EngineInputs
from jsbsim_parallel.models.propulsion_models.turbine import Turbine

from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.input_output.simulator_service import SimulatorService
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.input_output.model_loader import ModelLoader

class Propulsion(ModelBase, SimulatorService):
    def __init__(self, mass_balance: MassBalance, simulator_service: SimulatorService,
                 *, device: torch.device, batch_size: Optional[torch.Size] = None):
        super().__init__(simulator_service, device=device, batch_size=batch_size)
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.mass_balance = mass_balance
        self.simulator_service = simulator_service
        self.active_engine = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device) # -1: ALL, 0: Engine 1, 1: Engine 2 ...
        self.tankJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        self.DumpRate = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.RefuelRate = torch.tensor([6000.0], dtype=torch.float64, device=self.device).expand(*self.size, 1)
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
        self.ReadingEngine = False

    def FindFullPathName(self, path: str):
        name = super().FindFullPathName(path)
        if not self.ReadingEngine and not len(name) == 0:
            return name
        # TODO: Support more engine paths
        name = super().CheckPathName(os.path.join(self.simulator_service.GetFullAircraftPath(), "engine"), path)
        if len(name) > 0:
            return name
        return super().CheckPathName(self.simulator_service.GetEnginePath(), path)
    
    def Load(self, el: Element) -> bool:
        loader = ModelLoader(self)
        self.ReadingEngine = False
        fuelDensity = torch.full((*self.size, 1), 6.0, dtype=torch.float64, device=self.device)

        self.Name = "Propulsion Model: " + el.GetAttributeValue("name")

        if not super().Upload(el, True):
            return False
        
        tank_element = el.FindElement("tank")
        num_tanks = 0
        while tank_element is not None:
            tank = Tank(tank_element, num_tanks, device=self.device, batch_size=self.size)
            self.tanks.append(tank)
            num_tanks += 1
            if tank.GetType() == TankType.Fuel:
                fuelDensity = tank.GetDensity()
            elif tank.GetType() != TankType.Oxidizer:
                print("Unknown tank type")
                return False
            tank_element = el.FindNextElement("tank")
        
        self.ReadingEngine = True
        engine_element = el.FindElement("engine")
        numEngines = 0
        while engine_element is not None:
            if not loader.open(engine_element):
                return False
            thruster_element = engine_element.FindElement("thruster")
            if not thruster_element or not loader.open(thruster_element):
                    raise Exception("No thruster definition supplied with engine definition.")
            
            if engine_element.FindElement("turbine_engine") is not None:
                element = engine_element.FindElement("turbine_engine")
                engine = Turbine(self.mass_balance, 
                                 self.simulator_service,
                                 element, 
                                 numEngines, 
                                 self._in, 
                                 device=self.device, batch_size=self.size)
                self.engines.append(engine)
            else:
                print("Unsupported engine type")
                raise Exception("Unsupported engine type")
            numEngines += 1
            engine_element = el.FindNextElement("engine")
        
        #TODO: Bind properties
        self.CalculateTankInertias()
        if el.FindElement("dump-rate") is not None:
            self.DumpRate.fill_(el.FindElementValueAsNumberConvertTo("dump-rate", "LBS/MIN"))
        if el.FindElement("refuel-rate") is not None:
            self.RefuelRate.fill_(el.FindElementValueAsNumberConvertTo("refuel-rate", "LBS/MIN"))

        for engine in self.engines:
            engine.SetFuelDensity(fuelDensity)

        self.PostLoad(el)

        return True

    def GetNumEngines(self) -> int:
        return len(self.engines)
    
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