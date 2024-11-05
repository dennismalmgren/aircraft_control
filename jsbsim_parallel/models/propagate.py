from typing import Optional
from enum import Enum

import torch

from jsbsim_parallel.models.inertial import Inertial
from jsbsim_parallel.math.location import Location
from jsbsim_parallel.math.quaternion import Quaternion

class IntegratorType(Enum):
  NoIntegrator = 0
  RectEuler = 1
  Trapezoidal = 2
  AdamsBashforth2 = 3
  AdamsBashforth3 = 4
  AdamsBashforth4 = 5
  Buss1 = 6
  Buss2 = 7
  LocalLinearization = 8
  AdamsBashforth5 = 9

class PropagateInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Use batch_size for initialization if provided, else default to a single instance.
        size = batch_size if batch_size is not None else torch.Size([])

        # Angular acceleration in the body frame (expressed in the body frame)
        self.vPQRidot = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        
        # Linear acceleration in the body frame (expressed in the body frame)
        self.vUVWidot = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        
        # Earth rotation vector (expressed in the ECI frame)
        self.vOmegaPlanet = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        
        # Semi-major axis of the Earth (scalar, can be batch-processed if needed)
        self.SemiMajor = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        
        # Semi-minor axis of the Earth (scalar, can be batch-processed if needed)
        self.SemiMinor = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        
        # Gravitational parameter (scalar, can be batch-processed if needed)
        self.GM = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        
        # Time step (scalar, can be batch-processed if needed)
        self.DeltaT = torch.zeros(*size, 1, dtype=torch.float64, device=device)

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        self.vPQRidot = self.vPQRidot.to(device)
        self.vUVWidot = self.vUVWidot.to(device)
        self.vOmegaPlanet = self.vOmegaPlanet.to(device)
        
        self.SemiMajor = self.SemiMajor.to(device)
        self.SemiMinor = self.SemiMinor.to(device)
        self.GM = self.GM.to(device)
        self.DeltaT = self.DeltaT.to(device)


class VehicleState:
    def __init__(self, queue_length: int = 10, *, device, batch_size: Optional[torch.Size] = None):
        # Define batch size as torch.Size if given, otherwise it's a scalar
        size = batch_size if batch_size is not None else torch.Size([])

        # Define attributes as tensors, setting the initial shapes with batch_size
        # FGLocation equivalent (using 3D coordinates)
        self.vLocation = Location(batch_size = size, device = device)

        # Velocity vector of the vehicle with respect to ECEF, expressed in the body system
        self.vUVW = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # Angular velocity vector for the vehicle relative to ECEF, in body frame
        self.vPQR = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # Angular velocity vector for the vehicle body frame relative to ECI, in body frame
        self.vPQRi = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # Orientation of vehicle relative to local NED frame (quaternion)
        self.qAttitudeLocal = Quaternion(batch_size = size, device=device)  # Assuming quaternion [w, x, y, z]

        # Orientation of vehicle relative to inertial ECI frame (quaternion)
        self.qAttitudeECI = Quaternion(batch_size = size, device=device)  # Assuming quaternion [w, x, y, z]

        # Placeholder for quaternion derivative (not fully specified in provided code)
        self.vQtrndot = Quaternion(batch_size = size, device=device)

        # Inertial velocity and position vectors
        self.vInertialVelocity = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.vInertialPosition = torch.zeros(*size, 3, dtype=torch.float64, device=device)

        # Initialize deque replacements using tensors with a "queue" dimension
        self.dqPQRidot = torch.zeros(*size, queue_length, 3, dtype=torch.float64, device=device)
        self.dqUVWidot = torch.zeros(*size, queue_length, 3, dtype=torch.float64, device=device)
        self.dqInertialVelocity = torch.zeros(*size, queue_length, 3, dtype=torch.float64, device=device)
       #self.dqQtrndot = torch.zeros(*size, queue_length, 4, dtype=torch.float64, device=device)
       #TODO: deque of quaternions..

        # Store queue length for managing updates
        self.queue_length = queue_length

    def update_queue(self, queue_name: str, new_value: torch.Tensor):
        """Shift the specified queue tensor and add a new value at the end."""
        queue_tensor = getattr(self, queue_name)

        # Shift left (drop the oldest entry)
        queue_tensor = torch.cat((queue_tensor[..., 1:, :], new_value.unsqueeze(-2)), dim=-2)

        # Update the attribute with the new queue state
        setattr(self, queue_name, queue_tensor)

    def add_pqrdot(self, new_pqrdot: torch.Tensor):
        self.update_queue("dqPQRidot", new_pqrdot)

    def add_uvwdot(self, new_uvwdot: torch.Tensor):
        self.update_queue("dqUVWidot", new_uvwdot)

    def add_inertial_velocity(self, new_velocity: torch.Tensor):
        self.update_queue("dqInertialVelocity", new_velocity)

    # def add_qtrndot(self, new_qtrndot: torch.Tensor):
    #     self.update_queue("dqQtrndot", new_qtrndot) #TODO

class Propagate:
#    FGFDMExec* Executive
    def __init__(self, inertial: Inertial, device: torch.device, batch_size: Optional[torch.Size] = None):
        '''
        The constructor initializes several variables, and sets the initial set
        of integrators to use as follows:
        - integrator, rotational rate = Adams Bashforth 2
        - integrator, translational rate = Adams Bashforth 2
        - integrator, rotational position = Trapezoidal
        - integrator, translational position = Trapezoidal
        @param Executive a pointer to the parent executive object 
        '''
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self.inertial = inertial
        self._in = PropagateInputs(device, batch_size)
        self.VState = VehicleState(queue_length=5, device=device, batch_size=batch_size)
      

        # Earth position angle (?)
        self.epa = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        #
        self.Tec2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.Tb2ec = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        #local to body frame matrix copy for immediate local use
        self.Tl2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        #body to local frame matrix copy for immediate local use
        self.Tb2l = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # local to ECEF matrix copy for immediate local use
        self.Tl2ec = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # ECEF to local frame matrix copy for immediate local use
        self.Tec2l = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # ECEF to ECI frame matrix copy for immediate local use
        self.Tec2i = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # ECI to ECEF frame matrix copy for immediate local use
        self.Ti2ec = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # ECI to body frame rotation matrix
        self.Ti2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        # body to ECI frame rotation matrix
        self.Tb2i = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        
        self.Ti2l = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.Tl2i = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

        # Orbital parameters
        self.h = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Inclination = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.RightAscension = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Eccentricity = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.PerigeeArgument = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TrueAnomaly = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.ApoapsisRadius = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.PeriapsisRadius = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.OrbitalPeriod = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        
        self.Qec2b = Quaternion(device=device, batch_size=self.size)

        self.LocalTerrainVelocity = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.LocalTerrainAngularVelocity = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

        #also needs a property manager eventually.
        self.integrator_rotational_rate = IntegratorType.RectEuler
        self.integrator_translational_rate = IntegratorType.AdamsBashforth2
        self.integrator_rotational_position = IntegratorType.RectEuler
        self.integrator_translational_position = IntegratorType.AdamsBashforth3
        #bind() propertymanager acts as a service bus, tying addresses to functions, get/set.
        #we need to avoid that pattern to keep track of data transfers.

    def GetLocation(self) -> torch.Tensor:
        return self.VState.vLocation
    
    def run(holding: bool) -> bool:
        pass

    def init_model(self) -> bool:        
        #should call fgmodel::initmodel (base class fn)
        self.VState.vLocation.set_ellipse(self._in.SemiMajor, self._in.SemiMinor)
        init_altitude = torch.tensor([4.0], dtype=torch.float64, device=self.device).expand(*self.size, 1)
        self.inertial.set_altitude_AGL(self.VState.vLocation, init_altitude)

        #VState.dqPQRidot.resize(5, FGColumnVector3(0.0,0.0,0.0));
        #VState.dqUVWidot.resize(5, FGColumnVector3(0.0,0.0,0.0));
        #VState.dqInertialVelocity.resize(5, FGColumnVector3(0.0,0.0,0.0));
        #VState.dqQtrndot.resize(5, FGColumnVector3(0.0,0.0,0.0));

        self.integrator_rotational_rate = IntegratorType.RectEuler
        self.integrator_translational_rate = IntegratorType.AdamsBashforth2
        self.integrator_rotational_position = IntegratorType.RectEuler
        self.integrator_translational_position = IntegratorType.AdamsBashforth3

        self.epa = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        #reset to IC.
        return True

    def GetTl2b(self):
        return self.Tl2b
    
    def GetTb2l(self):
        return self.Tb2l
    
    def GetTec2l(self):
        return self.Tec2l

    def GetTec2b(self):
        return self.Tec2b
    
    def GetPQR(self):
        return self.VState.vPQR
    
    def GetUVW(self):
        return self.VState.vUVW
    

    def get_location(self):
        return self.VState.vLocation

    def GetAltitudeASL(self):
        return self.VState.vLocation.GetRadius() - self.VState.vLocation.GetSeaLevelRadius()

    def GetDistanceAGL(self):
        return self.inertial.get_altitude_AGL(self.VState.vLocation)
    
    def get_geod_latitude_deg(self):
        return self.VState.vLocation.GetGeodLatitudeDeg()

    def get_longitude_deg(self):
        return self.VState.vLocation.GetLongitudeDeg()
