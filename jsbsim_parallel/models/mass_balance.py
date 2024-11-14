from typing import Optional, List
from enum import IntEnum

import torch

from jsbsim_parallel.models.propagate import Propagate 
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.input_output.simulator_service import SimulatorService
from jsbsim_parallel.input_output.element import Element

class ShapeType(IntEnum):
    Unspecified = 0
    Tube = 1
    Cylinder = 2
    Sphere = 3
    Ball = 4

class PointMass:
    def __init__(self, w: torch.Tensor, vXYZ: torch.Tensor, 
                 *,
                 device: torch.device, batch_size: Optional[torch.Size]):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.ShapeType = ShapeType.Unspecified
        self.Location = vXYZ
        self.Weight = w #weight in pounds
        self.Radius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #radius in feet
        self.Length = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device) #length in feet
        self.Name = ""
        self.mPMInertia = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        self.units = UnitConversions.get_instance()

    def SetRadius(self, radius: torch.Tensor):
        self.Radius = radius
    
    def SetLength(self, length: torch.Tensor):
        self.Length = length

    def SetName(self, name: str):
        self.Name = name

    def SetPointMassShapeType(self, shapeType: ShapeType):
        self.ShapeType = shapeType

    def SetPointMassMoI(self, inertiaMatrix: torch.Tensor):
        self.mPMInertia = inertiaMatrix

    def CalculateShapeInertia(self):
        if self.ShapeType == ShapeType.Tube:
            self.mPMInertia[..., 0, 0]= (self.Weight / (self.units.SLUG_TO_LB)) * (self.Radius * self.Radius) # mr^2
            self.mPMInertia[..., 1, 1]= (self.Weight / (self.units.SLUG_TO_LB * 12)) * (6 * self.Radius * self.Radius + self.Length * self.Length)
            self.mPMInertia[..., 2, 2]= self.mPMInertia[..., 1, 1]
        elif self.ShapeType == ShapeType.Cylinder:
            self.mPMInertia[..., 0, 0]= (self.Weight / (self.units.SLUG_TO_LB * 2)) * (self.Radius * self.Radius) # mr^2
            self.mPMInertia[..., 1, 1]= (self.Weight / (self.units.SLUG_TO_LB * 12)) * (3 * self.Radius * self.Radius + self.Length * self.Length)
            self.mPMInertia[..., 2, 2]= self.mPMInertia[..., 1, 1]
        elif self.ShapeType == ShapeType.Sphere:
            self.mPMInertia[..., 0, 0]= (self.Weight / (self.units.SLUG_TO_LB * 3)) * (self.Radius * self.Radius * 2) # mr^2
            self.mPMInertia[..., 1, 1]= self.mPMInertia[..., 0, 0]
            self.mPMInertia[..., 2, 2]= self.mPMInertia[..., 0, 0]
        elif self.ShapeType == ShapeType.Ball:
            self.mPMInertia[..., 0, 0]= (self.Weight / (self.units.SLUG_TO_LB * 5)) * (self.Radius * self.Radius * 2) # mr^2
            self.mPMInertia[..., 1, 1]= self.mPMInertia[..., 0, 0]
            self.mPMInertia[..., 2, 2]= self.mPMInertia[..., 0, 0]
        else:
            pass


class MassBalanceInputs:
    def __init__(self, device: torch.device, batch_size: Optional[torch.Size] = None):
        # Define the size as a batch_size if provided, otherwise scalar
        size = batch_size if batch_size is not None else torch.Size([])

        self.GasMass = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.TanksWeight = torch.zeros(*size, 1, dtype=torch.float64, device=device)
        self.GasMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.GasInertia = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.TanksMoment = torch.zeros(*size, 3, dtype=torch.float64, device=device)
        self.TankInertia = torch.zeros(*size, 3, 3, dtype=torch.float64, device=device)
        self.WOW = torch.zeros(*size, 1, dtype=torch.bool, device=device)

class MassBalance(ModelBase):
    def __init__(self, propagate: Propagate, 
                                  simulator_service: SimulatorService,
    *, device, batch_size: Optional[torch.Size] = None):
        super().__init__(simulator_service, device=device, batch_size=batch_size)
        self.Name = "MassBalance"
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])

        self.propagate = propagate
        self.Weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.EmptyWeight = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.mass = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        
        self.vbaseXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=device)
        self.baseJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.mJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.mJinv = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.pmJ = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=device)
        self.units = UnitConversions.get_instance()
        self._in = MassBalanceInputs(device=device, batch_size=self.size)
        self.PointMasses: List[PointMass] = []

    def Load(self, document: Element) -> bool:
        self.Name = "Mass Properties Model: " + document.GetAttributeValue("name")

        if not super().Upload(document, True):
            return False
        
        inertias = self.ReadInertiaMatrix(document)
        self.SetAircraftBaseInertias(inertias)

        if document.FindElement("emptywt"):
            self.EmptyWeight.fill_(float(document.FindElementValueAsNumberConvertTo("emptywt", "LBS")))

        element = document.FindElement("location")
        while element is not None:
            element_name = element.GetAttributeValue("name")
            if element_name =="CG":
                self.vbaseXYZcg = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.size)
            element = document.FindNextElement("location")
        
        # Find all POINTMASS elements that descend from this METRICS branch of the
        # config file.
        element = document.FindElement("pointmass")
        while element is not None:
            self.AddPointMass(element)
            element = document.FindNextElement("pointmass")


        # TODO: Add Child FDM weights
        # need to investigate if F16 has child FDM's.

        self.Weight = self.EmptyWeight + self._in.TanksWeight + self.GetTotalPointMassWeight() + \
        self._in.GasMass * self.units.SLUG_TO_LB # + ChildFDMWeight

        self.Mass = self.units.LB_TO_SLUG * self.Weight

        super().PostLoad(document)
        return True
    
    def GetTotalPointMassWeight(self) -> torch.Tensor:
        total_weight = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        for pm in self.PointMasses:
            total_weight += pm.Weight
        return total_weight

    def AddPointMass(self, el: Element):
        loc_element = el.FindElement("location")
        pointmass_name = el.GetAttributeValue("name")
        if not loc_element:
            print("Pointmass " + pointmass_name + " has no location")
            raise Exception("Pointmass " + pointmass_name + " has no location")

        w = torch.tensor(el.FindElementValueAsNumberConvertTo("weight", "LBS"), dtype=torch.float64, device=self.device).expand(*self.size, 1)
        vXYZ = loc_element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.size)
        pm = PointMass(w, vXYZ, device=self.device, batch_size=self.size)
        pm.SetName(pointmass_name)

        form_element = el.FindElement("form")
        if form_element is not None:
            radius = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
            length = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
            shape = form_element.GetAttributeValue("shape")
            radius_element = form_element.FindElement("radius")
            length_element = form_element.FindElement("length")
            if radius_element is not None:
                radius = radius.fill_(radius_element.FindElementValueAsNumberConvertTo("FT"))
            if length_element is not None:
                length = length.fill_(length_element.FindElementValueAsNumberConvertTo("FT"))
            if shape == "tube":
                pm.SetPointMassShapeType(ShapeType.Tube)
                pm.SetRadius(radius)
                pm.SetLength(length)
                pm.CalculateShapeInertia()
            elif shape == "cylinder":
                pm.SetPointMassShapeType(ShapeType.Cylinder)
                pm.SetRadius(radius)
                pm.SetLength(length)
                pm.CalculateShapeInertia()
            elif shape == "sphere":
                pm.SetPointMassShapeType(ShapeType.Sphere)
                pm.SetRadius(radius)
                pm.CalculateShapeInertia()
            elif shape == "ball":
                pm.SetPointMassShapeType(ShapeType.Ball)
                pm.SetRadius(radius)
                pm.CalculateShapeInertia()
            #else:
            #    pass
        else:
            pm.SetPointMassShapeType(ShapeType.Unspecified)
            im = self.ReadInertiaMatrix(el)
            pm.SetPointMassMoI(im)

        # TODO: BIND PROPERTIES
        self.PointMasses.append(pm)

    def SetAircraftBaseInertias(self, inertias: torch.Tensor):
        self.baseJ = inertias

    def ReadInertiaMatrix(self, document: Element) -> torch.Tensor:
        # Initialize values to zero
        bixx = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        biyy = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        bizz = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        bixy = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        bixz = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        biyz = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

         # Update values if elements exist in document
        if document.FindElement("ixx"):
            bixx.fill_(document.FindElementValueAsNumberConvertTo("ixx", "SLUG*FT2"))
        if document.FindElement("iyy"):
            biyy.fill_(document.FindElementValueAsNumberConvertTo("iyy", "SLUG*FT2"))
        if document.FindElement("izz"):
            bizz.fill_(document.FindElementValueAsNumberConvertTo("izz", "SLUG*FT2"))
        if document.FindElement("ixy"):
            bixy.fill_(document.FindElementValueAsNumberConvertTo("ixy", "SLUG*FT2"))
        if document.FindElement("ixz"):
            bixz.fill_(document.FindElementValueAsNumberConvertTo("ixz", "SLUG*FT2"))
        if document.FindElement("iyz"):
            biyz.fill_(document.FindElementValueAsNumberConvertTo("iyz", "SLUG*FT2"))

        # Transform the inertia products from the structural frame to the body frame
        # and create the inertia matrix.
        if document.GetAttributeValue("negated_crossproduct_inertia") == "false":
            inertia_matrix = torch.stack([
                torch.stack([bixx,  bixy, -bixz], dim=-1),
                torch.stack([bixy,  biyy,  biyz], dim=-1),
                torch.stack([-bixz, biyz,  bizz], dim=-1)
            ], dim=-2)
        else:
            inertia_matrix = torch.stack([
                torch.stack([bixx, -bixy,  bixz], dim=-1),
                torch.stack([-bixy, biyy, -biyz], dim=-1),
                torch.stack([bixz, -biyz, bizz], dim=-1)
            ], dim=-2)
        return inertia_matrix
    
    def GetJinv(self) -> torch.Tensor:
        return self.mJinv
    
    def GetJ(self) -> torch.Tensor:
        return self.mJ
    
    # Conversion from the structural frame to the body frame.
    def StructuralToBody(self, r: torch.Tensor) -> torch.Tensor:
        '''
         Under the assumption that in the structural frame the:
        
         - X-axis is directed afterwards,
         - Y-axis is directed towards the right,
         - Z-axis is directed upwards,
        
         (as documented in http://jsbsim.sourceforge.net/JSBSimCoordinates.pdf)
         we have to subtract first the center of gravity of the plane which
         is also defined in the structural frame:
        
           FGColumnVector3 cgOff = r - vXYZcg;
        
         Next, we do a change of units:
        
           cgOff *= inchtoft;
        
         And then a 180 degree rotation is done about the Y axis so that the:
        
         - X-axis is directed forward,
         - Y-axis is directed towards the right,
         - Z-axis is directed downward.
        
         This is needed because the structural and body frames are 180 degrees apart.
        '''
        return torch.cat([
            (r[..., 0] - self.vXYZcg[..., 0]).unsqueeze(-1),
            (-r[..., 1] - self.vXYZcg[..., 1]).unsqueeze(-1),
            (-r[..., 2] - self.vXYZcg[..., 2]).unsqueeze(-1)
        ], dim=-1)* self.units.INCH_TO_FT
    
    def GetPointmassInertia(self, mass_sl: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        v = self.StructuralToBody(r)
        sv = mass_sl * v
        xx = sv[..., 0] * v[..., 0]
        yy = sv[..., 1] * v[..., 1]
        zz = sv[..., 2] * v[..., 2]
        xy = -sv[..., 0] * v[..., 1]
        xz = -sv[..., 0] * v[..., 2]
        yz = -sv[..., 1] * v[..., 2]
        result = torch.zeros(*self.size, 3, 3, dtype=torch.float64, device=self.device)
        result[..., 0, 0] = yy + zz
        result[..., 0, 1] = xy
        result[..., 0, 2] = xz
        result[..., 1, 0] = xy
        result[..., 1, 1] = xx + zz
        result[..., 1, 2] = yz
        result[..., 2, 0] = xz
        result[..., 2, 1] = yz
        result[..., 2, 2] = xx + yy
        return result
    
    def GetMass(self) -> torch.Tensor:
        return self.mass

    def GetXYZcg(self):
        return self.vXYZcg
    
    def GetEmptyWeight(self):
        return self.EmptyWeight
    
    def run(holding: bool) -> bool:
        if (holding):
            return True
        return False

    def init_model(self) -> bool:
        if not super().init_model():
            return False
        self.vLastXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        self.vDeltaXYZcg = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device)
        
        return True