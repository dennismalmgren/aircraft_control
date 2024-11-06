from typing import Optional, Dict, Any
from enum import IntEnum
import os

import torch



from .models import (
    Inertial,
    Propagate,
    Aerodynamics,
    Auxiliary,
    Accelerations,
    Aircraft,
    GroundReactions,
    ExternalReactions,
    StandardAtmosphere,
    BuoyantForces,
    Winds,
    UnitConversions,
    FCS,
    MassBalance,
    Propulsion
)

from jsbsim_parallel.input_output.xml_filereader import XMLFileReader
from jsbsim_parallel.input_output.element import Element

from jsbsim_parallel.models.model_base import EulerAngles
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

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

class FDMExec(ModelPathProvider):
    def __init__(self, device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.batch_size = batch_size if batch_size is not None else torch.Size([])
        self.dT = 1.0 / 120.0 #todo: source from somewhere
        UnitConversions.initialize(self.device)
        self.RootDir = ""
        self.AircraftPath = "aircraft"
        self.EnginePath = "engine"
        self.SystemsPath = "systems"
        self.allocate()
        self.modelLoaded = False
        #TODO: instance->Tie

    def GetFullAircraftPath(self):
        return self.FullAircraftPath
    
    def SetRootDir(self, root_dir: str):
        self.RootDir = root_dir

    def SetAircraftPath(self, path: str):
        self.AircraftPath = os.path.join(self.RootDir, path)

    def SetSystemsPath(self, path: str):
        self.SystemPath = os.path.join(self.RootDir, path)

    def SetEnginePath(self, path: str):
        self.EnginePath = os.path.join(self.RootDir, path)

    def LoadModel(self, model: str, addModelToPath:bool = True, batch_size: torch.Size = None) -> bool:
        self.batch_size = batch_size
        if addModelToPath:
            self.FullAircraftPath = os.path.join(self.AircraftPath, model)
        else:
            self.FullAircraftPath = self.AircraftPath

        self.aircraftCfgFileName = os.path.join(self.FullAircraftPath, model + ".xml")
        if self.modelLoaded:
            self.deallocate()
            self.allocate()

        reader = XMLFileReader()
        document = reader.load_xml_document(self.aircraftCfgFileName)
        self.ReadPrologue(document)

        # Process the planet element. This element is OPTIONAL.
        element = document.FindElement("planet")
        if element is not None:
            # Will always be None
            result = self.LoadPlanet(element)
            if not result:
                print("Invalid planet")
                return result

        # Process the metrics element. This element is REQUIRED.
        element = document.FindElement("metrics")
        if element is not None:
            result = self.aircraft.Load(element)
            if not result:
                print("Invalid aircraft")
                return result
        else:
            print("No metrics element")
            return False
        
        print('ok')

    def LoadPlanet(self, element: Element) -> bool:
        return True
    
    def ReadPrologue(self, element: Element):
        AircraftName = element.GetAttributeValue("name");
        self.aircraft.SetAircraftName(AircraftName)


    def deallocate(self):
        pass

    def allocate(self):
        self.inertial = Inertial(self.device, self.batch_size)
        self.propagate = Propagate(self.inertial, self.device, self.batch_size)
        #self.input
        self.atmosphere = StandardAtmosphere(self.device, self.batch_size)
        self.winds = Winds(self, device = self.device, batch_size = self.batch_size)
        self.systems = FCS(self.device, self.batch_size)
        self.mass_balance = MassBalance(self.propagate, self, device=self.device, batch_size=self.batch_size)
        self.auxiliary = Auxiliary(self.atmosphere, self.device, self.batch_size)
        self.propulsion = Propulsion(self.mass_balance, self, device = self.device, batch_size = self.batch_size) 
        self.aerodynamics = Aerodynamics(device = self.device, batch_size = self.batch_size)
        self.ground_reactions = GroundReactions(self, device=self.device, batch_size=self.batch_size)
        self.external_reactions = ExternalReactions(self.device, self.batch_size)
        self.buoyant_forces = BuoyantForces(self.device, self.batch_size)
        self.aircraft = Aircraft(self, device=self.device, batch_size=self.batch_size)
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
            ModelOrder.BuoyantForces: self.buoyant_forces,
            ModelOrder.Aircraft: self.aircraft,
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
        if model_id == ModelOrder.Propagate:
            self.propagate._in.vPQRidot = self.accelerations.GetPQRidot()
            self.propagate._in.vUVWidot = self.accelerations.GetUVWidot()
            self.propagate._in.DeltaT = self.dT
        elif model_id == ModelOrder.Input:
            pass
        elif model_id == ModelOrder.Inertial:
            self.inertial._in.Position = self.propagate.get_location()
        elif model_id == ModelOrder.Atmosphere:
            self.atmosphere._in.altitudeASL = self.propagate.GetAltitudeASL()
            self.atmosphere._in.GeodLatitudeDeg = self.propagate.GetGeodLatitudeDeg()
            self.atmosphere._in.LongitudeDeg = self.propagate.GetLongitudeDeg()
        elif model_id == ModelOrder.Winds:
            self.winds._in.AltitudeASL = self.propagate.GetAltitudeASL()
            self.winds._in.DistanceAGL = self.propagate.GetDistanceAGL()
            self.winds._in.Tl2b = self.propagate.GetTl2b()
            self.winds._in.Tw2b = self.auxiliary.GetTw2b()
            self.winds._in.V = self.auxiliary.GetVt()
            self.winds._in.totalDeltaT = self.dT * self.winds.GetRate()
        elif model_id == ModelOrder.Auxiliary:
            self.auxiliary._in.Pressure     = self.atmosphere.GetPressure()
            self.auxiliary._in.Density      = self.atmosphere.GetDensity()
            self.auxiliary._in.Temperature  = self.atmosphere.GetTemperature()
            self.auxiliary._in.SoundSpeed   = self.atmosphere.GetSoundSpeed()
            self.auxiliary._in.KinematicViscosity = self.atmosphere.GetKinematicViscosity()
            self.auxiliary._in.DistanceAGL  = self.propagate.GetDistanceAGL()
            self.auxiliary._in.Mass         = self.mass_balance.GetMass()
            self.auxiliary._in.Tl2b         = self.propagate.GetTl2b()
            self.auxiliary._in.Tb2l         = self.propagate.GetTb2l()
            self.auxiliary._in.vPQR         = self.propagate.GetPQR()
            self.auxiliary._in.vPQRi        = self.propagate.GetPQRi()
            self.auxiliary._in.vPQRidot     = self.accelerations.GetPQRidot()
            self.auxiliary._in.vUVW         = self.propagate.GetUVW()
            self.auxiliary._in.vUVWdot      = self.accelerations.GetUVWdot()
            self.auxiliary._in.vVel         = self.propagate.GetVel()
            self.auxiliary._in.vBodyAccel   = self.accelerations.GetBodyAccel()
            self.auxiliary._in.ToEyePt      = self.mass_balance.StructuralToBody(self.aircraft.GetXYZep())
            self.auxiliary._in.VRPBody      = self.mass_balance.StructuralToBody(self.aircraft.GetXYZvrp())
            self.auxiliary._in.RPBody       = self.mass_balance.StructuralToBody(self.aircraft.GetXYZrp())
            self.auxiliary._in.vFw          = self.aerodynamics.GetvFw()
            self.auxiliary._in.vLocation    = self.propagate.GetLocation()
            self.auxiliary._in.CosTht       = self.propagate.GetCosEuler(EulerAngles.Tht)
            self.auxiliary._in.SinTht       = self.propagate.GetSinEuler(EulerAngles.Tht)
            self.auxiliary._in.CosPhi       = self.propagate.GetCosEuler(EulerAngles.Phi)
            self.auxiliary._in.SinPhi       = self.propagate.GetSinEuler(EulerAngles.Phi)
            self.auxiliary._in.TotalWindNED = self.winds.GetTotalWindNED()
            self.auxiliary._in.TurbPQR      = self.winds.GetTurbPQR()
        elif model_id == ModelOrder.Systems:
            pass
        elif model_id == ModelOrder.Propulsion:
            self.propulsion._in.Pressure         = self.atmosphere.GetPressure()
            self.propulsion._in.PressureRatio    = self.atmosphere.GetPressureRatio()
            self.propulsion._in.Temperature      = self.atmosphere.GetTemperature()
            self.propulsion._in.DensityRatio     = self.atmosphere.GetDensityRatio()
            self.propulsion._in.Density          = self.atmosphere.GetDensity()
            self.propulsion._in.Soundspeed       = self.atmosphere.GetSoundSpeed()
            self.propulsion._in.TotalPressure    = self.auxiliary.GetTotalPressure()
            self.propulsion._in.Vc               = self.auxiliary.GetVcalibratedKTS()
            self.propulsion._in.Vt               = self.auxiliary.GetVt()
            self.propulsion._in.qbar             = self.auxiliary.Getqbar()
            self.propulsion._in.TAT_c            = self.auxiliary.GetTAT_C()
            self.propulsion._in.AeroUVW          = self.auxiliary.GetAeroUVW()
            self.propulsion._in.AeroPQR          = self.auxiliary.GetAeroPQR()
            self.propulsion._in.alpha            = self.auxiliary.Getalpha()
            self.propulsion._in.beta             = self.auxiliary.Getbeta()
            self.propulsion._in.TotalDeltaT      = self.dT * self.propulsion.GetRate()
            self.propulsion._in.ThrottlePos      = self.systems.GetThrottlePos()
            self.propulsion._in.MixturePos       = self.systems.GetMixturePos()
            self.propulsion._in.ThrottleCmd      = self.systems.GetThrottleCmd()
            self.propulsion._in.MixtureCmd       = self.systems.GetMixtureCmd()
            self.propulsion._in.PropAdvance      = self.systems.GetPropAdvance()
            self.propulsion._in.PropFeather      = self.systems.GetPropFeather()
            self.propulsion._in.H_agl            = self.propagate.GetDistanceAGL()
            self.propulsion._in.PQRi             = self.propagate.GetPQRi()
        elif model_id == ModelOrder.Aerodynamics:
            self.aerodynamics._in.Alpha     = self.auxiliary.Getalpha()
            self.aerodynamics._in.Beta      = self.auxiliary.Getbeta()
            self.aerodynamics._in.Qbar      = self.auxiliary.Getqbar()
            self.aerodynamics._in.Vt        = self.auxiliary.GetVt()
            self.aerodynamics._in.Tb2w      = self.auxiliary.GetTb2w()
            self.aerodynamics._in.Tw2b      = self.auxiliary.GetTw2b()
            self.aerodynamics._in.RPBody    = self.mass_balance.StructuralToBody(self.aircraft.GetXYZrp())
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
        elif model_id == ModelOrder.BuoyantForces:
            self.buoyant_forces._in.Density     = self.atmosphere.GetDensity()
            self.buoyant_forces._in.Pressure    = self.atmosphere.GetPressure()
            self.buoyant_forces._in.Temperature = self.atmosphere.GetTemperature()
            self.buoyant_forces._in.gravity     = self.inertial.GetGravity().norm(2, dim=-1, keepdim=True)
        elif model_id == ModelOrder.MassBalance:
            self.mass_balance._in.GasInertia  = self.buoyant_forces.GetGasMassInertia()
            self.mass_balance._in.GasMass     = self.buoyant_forces.GetGasMass()
            self.mass_balance._in.GasMoment   = self.buoyant_forces.GetGasMassMoment()
            self.mass_balance._in.TanksWeight = self.propulsion.GetTanksWeight()
            self.mass_balance._in.TanksMoment = self.propulsion.GetTanksMoment()
            self.mass_balance._in.TankInertia = self.propulsion.CalculateTankInertias()
            self.mass_balance._in.WOW         = self.ground_reactions.GetWOW()
        elif model_id == ModelOrder.Aircraft:
            self.aircraft._in.AeroForce     = self.aerodynamics.GetForces()
            self.aircraft._in.PropForce     = self.propulsion.GetForces()
            self.aircraft._in.GroundForce   = self.ground_reactions.GetForces()
            self.aircraft._in.ExternalForce = self.external_reactions.GetForces()
            self.aircraft._in.BuoyantForce  = self.buoyant_forces.GetForces()
            self.aircraft._in.AeroMoment    = self.aerodynamics.GetMoments()
            self.aircraft._in.PropMoment    = self.propulsion.GetMoments()
            self.aircraft._in.GroundMoment  = self.ground_reactions.GetMoments()
            self.aircraft._in.ExternalMoment = self.external_reactions.GetMoments()
            self.aircraft._in.BuoyantMoment = self.buoyant_forces.GetMoments()
        elif model_id == ModelOrder.Accelerations:
            self.accelerations._in.J        = self.mass_balance.GetJ()
            self.accelerations._in.Jinv     = self.mass_balance.GetJinv()
            self.accelerations._in.Ti2b     = self.propagate.GetTi2b()
            self.accelerations._in.Tb2i     = self.propagate.GetTb2i()
            self.accelerations._in.Tec2b    = self.propagate.GetTec2b()
            self.accelerations._in.Tec2i    = self.propagate.GetTec2i()
            self.accelerations._in.Moment   = self.aircraft.GetMoments()
            self.accelerations._in.GroundMoment  = self.ground_reactions.GetMoments()
            self.accelerations._in.Force    = self.aircraft.GetForces()
            self.accelerations._in.GroundForce   = self.ground_reactions.GetForces()
            self.accelerations._in.vGravAccel = self.inertial.GetGravity()
            self.accelerations._in.vPQRi    = self.propagate.GetPQRi()
            self.accelerations._in.vPQR     = self.propagate.GetPQR()
            self.accelerations._in.vUVW     = self.propagate.GetUVW()
            self.accelerations._in.vInertialPosition = self.propagate.GetInertialPosition()
            self.accelerations._in.DeltaT   = self.dT
            self.accelerations._in.Mass     = self.mass_balance.GetMass()
            self.accelerations._in.MultipliersList = self.ground_reactions.GetMultipliersList()
            self.accelerations._in.TerrainVelocity = self.propagate.GetTerrainVelocity()
            self.accelerations._in.TerrainAngularVel = self.propagate.GetTerrainAngularVelocity()
