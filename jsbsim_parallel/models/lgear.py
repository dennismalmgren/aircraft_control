from enum import IntEnum
from typing import Optional, List

import torch


class BrakeGroup(IntEnum):
    NoBrake = 0
    Left = 1
    Right = 2
    Center = 3
    Nose = 4
    Tail = 5
    NumBrakeGroups = 6

class SteerType(IntEnum):
    Steer = 0
    Fixed = 1
    Caster = 2

class ContactType(IntEnum):
    BOGEY = 0
    STRUCTURE = 1

class ReportType(IntEnum):
    NoReport = 0
    Takeoff = 1
    Landg = 2

class DampType(IntEnum):
    Linear = 0
    Square = 1

class FrictionType(IntEnum):
    Roll = 0
    Side = 1
    Dynamic = 2

class LGearInputs:
    def __init__(self, device='cpu', batch_size: Optional[torch.Size] = None):
        """
        Initializes the Inputs structure with placeholders for various input properties.

        :param device: Device on which tensors are stored ('cpu' or 'cuda').
        :param batch_size: Optional batch size for batched inputs.
        """
        # Set size as batch size or default to a single dimension
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device

        # Scalar variables with a single element in an extra dimension
        self.Vground = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.VcalibratedKts = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Temperature = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DistanceAGL = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DistanceASL = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TotalDeltaT = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.FCSGearPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.EmptyWeight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        # Boolean variables
        self.TakeoffThrottle = torch.tensor([False] * batch_size.numel() if batch_size else [False], 
                                            dtype=torch.bool, device=device)
        self.WOW = torch.tensor([False] * batch_size.numel() if batch_size else [False], 
                                dtype=torch.bool, device=device)

        # 3x3 matrices
        self.Tb2l = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.Tec2l = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.Tec2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

        # 3-element vectors
        self.PQR = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.UVW = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        
        # Location as a 3-element vector (assuming FGLocation is a 3-element vector)
        self.Location = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

        # List of doubles for BrakePos
        # Assuming that batch size applies here, each instance in the batch would have a list of brake positions
        # Initializing as an empty list here; actual values can be set later per batch instance if needed
        self.BrakePos: List[torch.Tensor] = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(batch_size.numel() if batch_size else 1)]
        #TODO: No Lists

class LGear:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        

        # Integer and static variables
        self.GearNumber = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)  # 3x3 identity matrix

        # Static matrices (Tb2s and Ts2b)
        self.Tb2s = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)  # 3x3 identity matrix
        self.Ts2b = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)  # 3x3 identity matrix

        # 3x3 Matrix for TGear
        self.mTGear = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)

        # 3-element vectors
        self.vLocalGear = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vWhlVelVec = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vGroundWhlVel = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vGroundNormal = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)

        # Table and Function placeholders
        self.ForceY_Table = None  # Placeholder for FGTable instance
        self.fStrutForce = None  # Placeholder for FGFunction instance

        # Double scalar values
        self.SteerAngle = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.kSpring = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.bDamp = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.bDampRebound = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.compressLength = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.compressSpeed = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.staticFCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.dynamicFCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.rollingFCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Stiffness = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Shape = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Peak = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Curvature = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.BrakeFCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.maxCompLen = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.SinkRate = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.GroundSpeed = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TakeoffDistanceTraveled = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.TakeoffDistanceTraveled50ft = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.LandingDistanceTraveled = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.MaximumStrutForce = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.StrutForce = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.MaximumStrutTravel = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.FCoeff = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.WheelSlip = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.GearPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.staticFFactor = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        self.rollingFFactor = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        self.maximumForce = torch.full(*self.size, float('inf'), dtype=torch.float64, device=device)
        self.bumpiness = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.AGL = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.maxSteerAngle = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        # Boolean values
        self.isSolid = torch.tensor([True] * batch_size.numel() if batch_size else [True], dtype=torch.bool, device=device)
        self.WOW = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.lastWOW = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.FirstContact = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.StartedGroundRun = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.LandingReported = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.TakeoffReported = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.ReportEnable = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.isRetractable = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.Castered = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.StaticFriction = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)
        self.useFCSGearPos = torch.tensor([False] * batch_size.numel() if batch_size else [False], dtype=torch.bool, device=device)

        # Enums
        self.eBrakeGrp = BrakeGroup.NoBrake
        self.eContactType = ContactType.BOGEY
        self.eSteerType = SteerType.Fixed
        self.eDampType = DampType.Linear
        self.eDampTypeRebound = DampType.Linear

        # Lagrange multiplier and other objects
        self.LMultiplier = [torch.zeros(3, dtype=torch.float64, device=device) for _ in range(3)]
        self.GroundReactions = None  # Placeholder for FGGroundReactions instance

        self._in = LGearInputs(device=device, batch_size=self.size)
    
    def IsBogey(self):
        return self.eContactType == ContactType.BOGEY

    def GetWOW(self):
        return self.WOW