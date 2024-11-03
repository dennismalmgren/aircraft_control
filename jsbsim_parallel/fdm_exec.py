from typing import Optional
import torch


from .models import (
    Inertial,
    Propagate,
    Auxiliary,
    Accelerations,
    StandardAtmosphere,
    Winds,
    UnitConversions,
    FGFCS,
    MassBalance,
    Propulsion
)

class FDMExec:
    def __init__(self, device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size
        self.dT = 1.0 / 120.0 #todo: source from somewhere
        UnitConversions.initialize(self.device)

    def allocate(self):

        self.inertial = Inertial(self.device, self.batch_size)
        self.propagate = Propagate(self.inertial, self.device, self.batch_size)
        #self.input
        self.atmosphere = StandardAtmosphere(self.device, self.batch_size)
        self.winds = Winds(device = self.device, batch_size = self.batch_size)
        self.fgfcs = FGFCS(self.device, self.batch_size)
        self.mass_balance = MassBalance(self.propagate, device=self.device, batch_size=self.batch_size)
        self.auxiliary = Auxiliary(self.device, self.batch_size)
        self.propulsion = Propulsion(device = self.device, batch_size = self.batch_size) #todo:
        #self.aerodynamics
        #self.ground_reactions
        #self.external_reactions
        #self.buoyantforces
        #self.aircraft
        self.accelerations = Accelerations(self.device, self.batch_size)
        #self.output
        self.models = [self.inertial, self.propagate, self.atmosphere, self.auxiliary, self.accelerations]
        self.load_planet_constants()
        self.initialize_models()

    def load_planet_constants(self):
        self.propagate._in.vOmegaPlanet = self.inertial.omega_planet() #note, should be cloning (?) but these are more or less constants, now.
        self.accelerations._in.vOmegaPlanet = self.inertial.omega_planet() #note, should be cloning (?) but these are more or less constants, now.
        self.propagate._in.SemiMajor = self.inertial.semi_major()
        self.propagate._in.SemiMinor = self.inertial.semi_minor()
        self.propagate._in.GM = self.inertial.GM()
        self.auxiliary._in.StandardGravity = self.inertial.standard_gravity()
        self.auxiliary._in.StdDaySLsoundspeed  = self.atmosphere.StdDaySLsoundspeed

    def initialize_models(self):
        #inertial
        self.inertial._in.Position = self.propagate.get_location()
        self.inertial.init_model()
        #propagate
        self.propagate._in.vPQRidot = self.accelerations.get_pqr_idot()
        self.propagate._in.vUVWidot = self.accelerations.get_uvw_idot()
        self.propagate.init_model()
        #input
        #do not initialize
        #atmosphere
        self.atmosphere._in.altitudeASL = self.propagate.get_altitude_ASL()
        self.atmosphere._in.geod_latitude_deg = self.propagate.get_geod_latitude_deg()
        self.atmosphere._in.longitude_deg = self.propagate.get_longitude_deg()
        #self.atmosphere.init_model()
        #winds
        self.winds._in.AltitudeASL = self.propagate.get_altitude_ASL()
        self.winds._in.DistanceAGL = self.propagate.get_distance_AGL()
        self.winds._in.Tl2b = self.propagate.get_Tl2b()
        self.winds._in.Tw2b = self.auxiliary.get_Tw2b()
        self.winds._in.V = self.auxiliary.get_Vt()
        self.winds._in.totalDeltaT = self.dT * self.winds.get_rate()
        self.winds.init_model()
        #fgfcs
        # Dynamic inputs come into the components that FCS manages through properties
        #propulsion
        self.propulsion._in.Pressure = self.atmosphere.get_pressure()
        self.propulsion._in.PressureRatio = self.atmosphere.get_pressure_ratio()
        self.propulsion._in.Temperature = self.atmosphere.get_temperature()
        self.propulsion._in.DensityRatio = self.atmosphere.get_density_ratio()
        #self.propulsion._in.Density = self.atmosphere.get_density()
        #self.propulsion._in.Soundspeed = self.atmosphere.get_soundspeed()
#  Propulsion->in.Pressure         = Atmosphere->GetPressure();
#     Propulsion->in.PressureRatio    = Atmosphere->GetPressureRatio();
#     Propulsion->in.Temperature      = Atmosphere->GetTemperature();
#     Propulsion->in.DensityRatio     = Atmosphere->GetDensityRatio();
#     Propulsion->in.Density          = Atmosphere->GetDensity();
#     Propulsion->in.Soundspeed       = Atmosphere->GetSoundSpeed();

        
            
