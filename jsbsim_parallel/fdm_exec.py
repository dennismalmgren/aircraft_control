from typing import Optional, Dict, Any
from enum import IntEnum

import torch


from .models import (
    Inertial,
    Propagate,
    Aerodynamics,
    Auxiliary,
    Accelerations,
    GroundReactions,
    ExternalReactions,
    StandardAtmosphere,
    Winds,
    UnitConversions,
    FCS,
    MassBalance,
    Propulsion
)

# This list of enums is very important! The order in which models are listed
# here determines the order of execution of the models.
#
# There are some conditions that need to be met :
# 1. FCS can request mass geometry changes via the inertia/pointmass-*
#    properties so it must be executed before MassBalance
# 2. MassBalance must be executed before Propulsion, Aerodynamics,
#    GroundReactions, ExternalReactions and BuoyantForces to ensure that
#    their moments are computed with the updated CG position.
class ModelOrder(IntEnum):
    Propagate = 0
    Input = 1
    Inertial = 2
    Atmosphere = 3
    Winds = 4
    Systems = 5
    MassBalance = 6
    Auxiliary = 7
    Propulsion = 8
    Aerodynamics = 9
    GroundReactions = 10
    ExternalReactions = 11
    BuoyantForces = 12
    Aircraft = 13
    Accelerations = 14
    Output = 15
    NumStandardModels = 16

class FDMExec:
    def __init__(self, device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size if batch_size is not None else torch.Size([])
        self.dT = 1.0 / 120.0 #todo: source from somewhere
        UnitConversions.initialize(self.device)

    def allocate(self):
        self.inertial = Inertial(self.device, self.batch_size)
        self.propagate = Propagate(self.inertial, self.device, self.batch_size)
        #self.input
        self.atmosphere = StandardAtmosphere(self.device, self.batch_size)
        self.winds = Winds(device = self.device, batch_size = self.batch_size)
        self.systems = FCS(self.device, self.batch_size)
        self.mass_balance = MassBalance(self.propagate, device=self.device, batch_size=self.batch_size)
        self.auxiliary = Auxiliary(self.atmosphere, self.device, self.batch_size)
        self.propulsion = Propulsion(device = self.device, batch_size = self.batch_size) 
        self.aerodynamics = Aerodynamics(device = self.device, batch_size = self.batch_size)
        self.ground_reactions = GroundReactions(self.device, self.batch_size)
        self.external_reactions = ExternalReactions(self.device, self.batch_size)
        #self.buoyantforces
        #self.aircraft
        self.accelerations = Accelerations(self.device, self.batch_size)
        #self.output
        self.models: Dict[ModelOrder, Any] = {
            ModelOrder.Inertial: self.inertial,
            ModelOrder.Propagate: self.propagate,
#            ModelOrder.Input: self.input,
            ModelOrder.Atmosphere: self.atmosphere,
            ModelOrder.Winds: self.winds,
            ModelOrder.Systems: self.systems,
            ModelOrder.MassBalance: self.mass_balance,
            ModelOrder.Auxiliary: self.auxiliary,
            ModelOrder.Propulsion: self.propulsion,
            ModelOrder.Aerodynamics: self.aerodynamics,
            ModelOrder.GroundReactions: self.ground_reactions,
            ModelOrder.ExternalReactions: self.external_reactions,
            #ModelOrder.BuoyantForces: self.buoyantforces,
            #ModelOrder.Aircraft: self.aircraft,
            ModelOrder.Accelerations: self.accelerations,
            #ModelOrder.Output: self.output
        }
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
        for model_id in ModelOrder:
            if model_id in self.models:
                self.load_inputs(model_id)
                self.models[model_id].init_model()
    
    def load_inputs(self, model_id):
        if model_id == ModelOrder.Inertial:
            self.inertial._in.Position = self.propagate.get_location()
        elif model_id == ModelOrder.Propagate:
            self.propagate._in.vPQRidot = self.accelerations.get_pqr_idot()
            self.propagate._in.vUVWidot = self.accelerations.get_uvw_idot()
        elif model_id == ModelOrder.Input:
            pass
        elif model_id == ModelOrder.Atmosphere:
            self.atmosphere._in.altitudeASL = self.propagate.GetAltitudeASL()
            self.atmosphere._in.geod_latitude_deg = self.propagate.get_geod_latitude_deg()
            self.atmosphere._in.longitude_deg = self.propagate.get_longitude_deg()
        elif model_id == ModelOrder.Winds:
            self.winds._in.AltitudeASL = self.propagate.GetAltitudeASL()
            self.winds._in.DistanceAGL = self.propagate.GetDistanceAGL()
            self.winds._in.Tl2b = self.propagate.GetTl2b()
            self.winds._in.Tw2b = self.auxiliary.GetTw2b()
            self.winds._in.V = self.auxiliary.GetVt()
            self.winds._in.totalDeltaT = self.dT * self.winds.GetRate()
        elif model_id == ModelOrder.Systems:
            pass
        elif model_id == ModelOrder.Propulsion:
            self.propulsion._in.Pressure = self.atmosphere.get_pressure()
            self.propulsion._in.PressureRatio = self.atmosphere.get_pressure_ratio()
            self.propulsion._in.Temperature = self.atmosphere.GetTemperature()
            self.propulsion._in.DensityRatio = self.atmosphere.get_density_ratio()
            self.propulsion._in.Density = self.atmosphere.GetDensity()
            self.propulsion._in.Soundspeed = self.atmosphere.GetSoundspeed()
        elif model_id == ModelOrder.Aerodynamics:
            self.aerodynamics._in.Alpha     = self.auxiliary.Getalpha()
            self.aerodynamics._in.Beta      = self.auxiliary.Getbeta()
            self.aerodynamics._in.Qbar      = self.auxiliary.Getqbar()
            self.aerodynamics._in.Vt        = self.auxiliary.GetVt()
            self.aerodynamics._in.Tb2w      = self.auxiliary.GetTb2w()
            self.aerodynamics._in.Tw2b      = self.auxiliary.GetTw2b()
            #self.aerodynamics._in.RPBody    = self.mass_balance.StructuralToBody(self.aircraft.GetXYZrp())
        elif model_id == ModelOrder.GroundReactions:
            # // There are no external inputs to this model.
            self.ground_reactions._in.Vground         = self.auxiliary.GetVground()
            self.ground_reactions._in.VcalibratedKts  = self.auxiliary.GetVcalibratedKTS()
            self.ground_reactions._in.Temperature     = self.atmosphere.GetTemperature()
            self.ground_reactions._in.TakeoffThrottle = self.systems.GetThrottlePos(0) > 0.90 \
                                                        if len(self.systems.GetThrottlePos()) > 0 \
                                                            else torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
            self.ground_reactions._in.BrakePos        = self.systems.GetBrakePos()
            self.ground_reactions._in.FCSGearPos      = self.systems.GetGearPos()
            self.ground_reactions._in.EmptyWeight     = self.mass_balance.GetEmptyWeight()
            self.ground_reactions._in.Tb2l            = self.propagate.GetTb2l()
            self.ground_reactions._in.Tec2l           = self.propagate.GetTec2l()
            self.ground_reactions._in.Tec2b           = self.propagate.GetTec2b()
            self.ground_reactions._in.PQR             = self.propagate.GetPQR()
            self.ground_reactions._in.UVW             = self.propagate.GetUVW()
            self.ground_reactions._in.DistanceAGL     = self.propagate.GetDistanceAGL()
            self.ground_reactions._in.DistanceASL     = self.propagate.GetAltitudeASL()
            self.ground_reactions._in.TotalDeltaT     = self.dT * self.ground_reactions.GetRate()
            self.ground_reactions._in.WOW             = self.ground_reactions.GetWOW() #check consistency
            self.ground_reactions._in.Location        = self.propagate.GetLocation()
            self.ground_reactions._in.vXYZcg          = self.mass_balance.GetXYZcg()
        elif model_id == ModelOrder.ExternalReactions:
            pass
            
