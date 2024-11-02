from typing import Optional

import torch
from jsbsim_parallel.models.unit_conversions import UnitConversions
#note: simulations should probably be done in float64

class AtmosphereInputs:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        size = batch_size if batch_size is not None else torch.Size([])
        self.device = device

        self.altitudeASL = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.geod_latitude_deg = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.longitude_deg = torch.zeros(*size, 1, dtype=torch.float64, device=device)


class StandardAtmosphere:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        units = UnitConversions.get_instance()
        #todo: unused (yet)
        size = batch_size if batch_size is not None else torch.Size([])

        self.SHRatio = torch.tensor(1.4, dtype=torch.float64, device=device)
        # Gas constant
        self.R = torch.tensor(8.31432, dtype=torch.float64, device=self.device)
        self.thousand = torch.tensor(1000.0, dtype=torch.float64, device=self.device)
        # Universal gas constant - ft*lbf/R/mol
        self.Rstar = self.R * units.KG_TO_SLUG / units.KelvinToRankine(units.FT_TO_M * units.FT_TO_M)
        # Relative molecular mass of components in dry air
        self.Rmm = torch.tensor(28.9645, dtype=torch.float64, device=self.device)
        # Mean molecular weight for air - slug/mol
        self.Mair = self.Rmm * units.KG_TO_SLUG / self.thousand

        self.Reng0 = self.Rstar / self.Mair
        self.Reng = self.Reng0
        self.StdDaySLtemperature = torch.tensor(518.67, dtype=torch.float64, device=self.device)
        self.StdDaySLsoundspeed = torch.sqrt(self.SHRatio*self.Reng0*self.StdDaySLtemperature)
        self._in = AtmosphereInputs(device=self.device, batch_size=batch_size)
