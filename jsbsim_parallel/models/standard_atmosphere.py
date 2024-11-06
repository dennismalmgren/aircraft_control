from typing import Optional

import torch
from jsbsim_parallel.models.unit_conversions import UnitConversions
#note: simulations should probably be done in float64
from jsbsim_parallel.math.table_1d import Table1D

class AtmosphereInputs:
    def __init__(self, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        size = batch_size if batch_size is not None else torch.Size([])
        self.device = device

        self.altitudeASL = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.GeodLatitudeDeg = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.LongitudeDeg = torch.zeros(*size, 1, dtype=torch.float64, device=device)


class StandardAtmosphere:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        #todo: unused (yet)
        self.size = batch_size if batch_size is not None else torch.Size([])
        units = UnitConversions.get_instance()

        #base class constants
        self.StdDaySLtemperature = torch.tensor(518.67, dtype=torch.float64, device=self.device)
        self.StdDaySLpressure = torch.tensor(2116.228, dtype=torch.float64, device=self.device)
        self.SHRatio = torch.tensor(1.4, dtype=torch.float64, device=device)
        
        # Relative molecular mass of components in dry air
        self.Rmm = torch.tensor(28.9645, dtype=torch.float64, device=self.device)

        # Gas constant
        self.R = torch.tensor(8.31432, dtype=torch.float64, device=self.device)
        # Universal gas constant - ft*lbf/R/mol
        self.Rstar = self.R * units.KG_TO_SLUG / units.KelvinToRankine(units.FT_TO_M * units.FT_TO_M)
        # Mean molecular weight for air - slug/mol
        thousand = torch.tensor(1000.0, dtype=torch.float64, device=self.device)
        self.Mair = self.Rmm * units.KG_TO_SLUG / thousand

        self.Reng0 = self.Rstar / self.Mair
        self.g0 = 9.80665 / units.FT_TO_M
        self.Rdry = self.Rstar / self.Mair
        #base class variable list
        self.Temperature = torch.tensor(1.8, dtype=torch.float64, device=self.device)
        self.Pressure = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.SLpressure: torch.Tensor = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        self.Density: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.SLdensity: torch.Tensor = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        self.SLtemperature: torch.Tensor = torch.tensor(1.8, dtype=torch.float64, device=self.device)
        self.SLsoundspeed: torch.Tensor = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        self.Soundspeed: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.KinematicViscosity: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.Viscosity: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        # base class init
        self.StdDaySLsoundspeed = torch.sqrt(self.SHRatio*self.Reng0*self.StdDaySLtemperature)
        
        # constructor initializer list
        self.StdSLpressure = self.StdDaySLpressure.clone()
        self.TemperatureBias = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.TemperatureDeltaGradient = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        #     VaporMassFraction(0.0),
        #     SaturatedVaporPressure(StdDaySLpressure)
        #constructor

        std_atmos_temperature_table_data = torch.tensor([[0.0000, 518.67],
                                                         [36089.2388, 389.97],
                                                         [65616.7979, 389.97],
                                                         [104986.8766, 411.57],
                                                         [154199.4751, 487.17],
                                                         [167322.8346, 487.17],
                                                         [232939.6325, 386.37],
                                                         [278385.8268, 336.5028],
                                                         [298556.4304, 336.5028],
                                                         ], dtype=torch.float64,
                                                         device=self.device)
        self.std_atmos_temperature_table =  Table1D(N=9, 
                                                    data = std_atmos_temperature_table_data,
                                                    device = self.device)
        max_vapor_mass_fraction_table_data = torch.tensor([[ 0.0000, 35000.],
                                                           [3280.8399, 31000.],
                                                           [6561.6798, 28000.],
                                                           [13123.3596, 22000.],
                                                           [19685.0394,  8900.],
                                                           [26246.7192,  4700.],
                                                           [32808.3990,  1300.],
                                                           [39370.0787,   230.],
                                                           [45931.7585,    48.],
                                                           [52493.4383,    38.],], dtype=torch.float64,
                                                           device = self.device)
        self.max_vapor_mass_fraction =  Table1D(N=10, 
                                                    data = max_vapor_mass_fraction_table_data,
                                                    device = self.device)
        self.LapseRates = torch.zeros(len(self.std_atmos_temperature_table), 
                                               dtype=torch.float64, device=self.device)
        self.calculate_lapse_rates()
        self.StdLapseRates = self.LapseRates.clone()
        
        self.GradientFadeoutAltitude = self.std_atmos_temperature_table[-1, 0]
        self.PressureBreakpoints = torch.zeros(len(self.std_atmos_temperature_table), 
                                               dtype=torch.float64, device=self.device)
        self.calculate_pressure_breakpoints(self.StdSLpressure)

        self.StdPressureBreakpoints = self.PressureBreakpoints.clone()
        self.StdSLtemperature = self.std_atmos_temperature_table[0, 1]
        self.StdSLdensity = self.StdSLpressure / (self.Rdry * self.StdSLtemperature)
        self.StdDensityBreakpoints = torch.zeros(len(self.StdPressureBreakpoints),
                                                 dtype=torch.float64, device=self.device)
        self.calculate_density_breakpoints()
        self.StdSLsoundspeed = torch.sqrt(self.SHRatio*self.Rdry*self.StdSLtemperature)

        # self.Reng = self.Reng0
        # self.StdDaySLtemperature = torch.tensor(518.67, dtype=torch.float64, device=self.device)
        # self.StdDaySLsoundspeed = torch.sqrt(self.SHRatio*self.Reng0*self.StdDaySLtemperature)
        self._in = AtmosphereInputs(device=self.device, batch_size=batch_size)
        # self.earth_radius = 6356766.0 / units.FT_TO_M
        

        '''
                               GeoPot Alt    Temp       GeoPot Alt  GeoMet Alt
                                   (ft)      (deg R)        (km)        (km)
                              -----------   --------     ----------  ----------
                           <<      0.0000 << 518.67  //    0.000       0.000
                           <<  36089.2388 << 389.97  //   11.000      11.019
                           <<  65616.7979 << 389.97  //   20.000      20.063
                           << 104986.8766 << 411.57  //   32.000      32.162
                           << 154199.4751 << 487.17  //   47.000      47.350
                           << 167322.8346 << 487.17  //   51.000      51.413
                           << 232939.6325 << 386.37  //   71.000      71.802
                           << 278385.8268 << 336.5028  // 84.852      86.000
                           << 298556.4304 << 336.5028
        '''

        
        #self.lapse_rates = torch.zeros(len(self.std_atmos_temperature_table), dtype=torch.float64, device=self.device)
        
    def calculate_density_breakpoints(self):
        for i in range(len(self.StdPressureBreakpoints)):
            self.StdDensityBreakpoints[i] = self.StdPressureBreakpoints[i] / (self.Rdry * self.std_atmos_temperature_table[i, 1])

    def calculate_pressure_breakpoints(self, sl_pressure: torch.Tensor):
        self.PressureBreakpoints[0] = sl_pressure
        for i in range(len(self.std_atmos_temperature_table) - 1):
            base_temp = self.std_atmos_temperature_table[i + 1, 1]
            base_alt = self.std_atmos_temperature_table[i + 1, 0]
            upper_alt = self.std_atmos_temperature_table[i + 1, 0]
            delta_alt = upper_alt - base_alt
            tmb = base_temp + self.TemperatureBias * (self.GradientFadeoutAltitude - base_alt) * \
                    self.TemperatureDeltaGradient                
            if self.LapseRates[i] != 0.0: #todo: tensor
                lmb = self.LapseRates[i]
                exp = self.g0 / (self.Rdry * lmb)
                factor = tmb / (tmb + lmb * delta_alt)
                self.PressureBreakpoints[i + 1] = self.PressureBreakpoints[i] * torch.pow(factor, exp)
            else:
                self.PressureBreakpoints[i + 1] = self.PressureBreakpoints[i] * torch.exp(-self.g0 * delta_alt / (self.Rdry * lmb))
            
    def calculate_lapse_rates(self):
        for base_height_index in range(0, len(self.std_atmos_temperature_table) - 1):
            t0 = self.std_atmos_temperature_table[base_height_index, 1]
            t1 = self.std_atmos_temperature_table[base_height_index + 1, 1]
            h0 = self.std_atmos_temperature_table[base_height_index, 0]
            h1 = self.std_atmos_temperature_table[base_height_index + 1, 0]
            self.LapseRates[base_height_index] = (t1 - t0) / (h1 - h0) - self.TemperatureDeltaGradient

    def init_model(self) -> bool:
        #base class init_model
#   SLtemperature = Temperature = StdDaySLtemperature;
#   SLpressure = Pressure = StdDaySLpressure;
#   SLdensity = Density = Pressure/(Reng*Temperature);
#   SLsoundspeed = Soundspeed = StdDaySLsoundspeed;
#   Calculate(0.0);

        self.GradientFadeoutAltitude = self.std_atmos_temperature_table[-1]
        
        self.TemperatureDeltaGradient = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.TemperatureBias = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.LapseRates.copy_(self.StdLapseRates)


        self.PressureBreakpoints = self.StdPressureBreakpoints

        self.SLpressure.copy_(self.StdSLpressure)
        self.SLtemperature.copy_(self.StdSLtemperature)
        self.SLdensity.copy_(self.StdSLdensity)
        self.SLsoundspeed.copy_(self.StdSLsoundspeed)

#        self.Calculate(0.0)

        return True

    def geo_potential_altitude(self, geomet_alt: torch.Tensor) -> torch.Tensor:
        return (geomet_alt * self.earth_radius) / (self.earth_radius + geomet_alt)
    
    def GetPressure(self) -> torch.Tensor:
        return self.Pressure
    
    def GetPressureRatio(self) -> torch.Tensor:
        return self.Pressure / self.SLpressure

    def GetTemperature(self) -> torch.Tensor:
        return self.Temperature

    def GetDensityRatio(self) -> torch.Tensor:
        return self.Density / self.SLdensity
    
    def GetDensity(self) -> torch.Tensor:
        return self.Density
    
    def GetSoundSpeed(self) -> torch.Tensor:
        return self.Soundspeed
    
    def GetKinematicViscosity(self) -> torch.Tensor:
        return self.KinematicViscosity
    
    #     self.propulsion._in.DensityRatio = self.atmosphere.get_density_ratio()
    #     self.propulsion._in.Density = self.atmosphere.get_density()
    #     self.propulsion._in.Soundspeed = self.atmosphere.get_soundspeed()