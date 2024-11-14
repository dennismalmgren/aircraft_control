from typing import Optional
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.unit_conversions import UnitConversions

class TankType(IntEnum):
    Unknown = 0
    Fuel = 1
    Oxidizer = 2

class GrainType(IntEnum):
    Unknown = 1
    Cylindrical = 2
    EndBurning = 3
    Function = 4

class Tank:
    def __init__(self, el: Element, tank_number: int, *, device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.units = UnitConversions.get_instance()
        self.TankNumber = tank_number
        self.Area = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.Density = torch.full((*self.size, 1), 6.6, dtype=torch.float64, device=self.device)
        self.InitialTemperature = torch.full((*self.size, 1), -9999.0, dtype=torch.float64, device=self.device)
        self.Temperature = torch.full((*self.size, 1), -9999.0, dtype=torch.float64, device=self.device)
        self.Ixx = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Iyy = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Izz = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.InertiaFactor = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.Radius = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Contents = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.InitialContents = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Standpipe = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Length = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.InnerRadius = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.ExternalFlow = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.InitialStandpipe = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Capacity = torch.full((*self.size, 1), 0.00001, dtype=torch.float64, device=self.device)
        self.UnusableVol = torch.full((*self.size, 1), 0.0, dtype=torch.float64, device=self.device)
        self.Priority = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.InitialPriority = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.vXYZ = torch.zeros(*self.size, 3, dtype=torch.float64, device=self.device) #initialized further down
        #self.vXYZ_drain = torch.zeros(self.size, 3, dtype=torch.float64, device=self.device) #initialized further down

        self.ixx_unit = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.iyy_unit = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.izz_unit = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.grainType = GrainType.Unknown

        g_type = el.GetAttributeValue("type")
        if g_type == "FUEL": 
            self.Type = TankType.Fuel
        elif g_type == "OXIDIZER":
            self.Type = TankType.Oxidizer
        else:
            self.Type = TankType.Unknown
        
        self.Name = el.GetAttributeValue("name")

        element = el.FindElement("location")
        if element is not None:
            self.vXYZ = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.size)
        else:
            print("No location element found for tank")
        
        self.vXYZ_drain = self.vXYZ.clone()  # Set initial drain location to initial tank CG

        element = el.FindElement("drain_location")
        if element is not None:
            self.vXYZ_drain = element.FindElementTripletConvertTo("IN", device=self.device, batch_size=self.size)

        if el.FindElement("radius") is not None:
            self.Radius.fill_(el.FindElementValueAsNumberConvertTo("radius", "IN"))

        if el.FindElement("inertia_factor") is not None:
            self.InertiaFactor.fill_(el.FindElementValueAsNumber("inertia_factor"))

        if el.FindElement("capacity") is not None:
            self.Capacity.fill_(el.FindElementValueAsNumberConvertTo("capacity", "LBS"))

        if el.FindElement("contents") is not None:
            initial_contents = el.FindElementValueAsNumberConvertTo("contents", "LBS")
            self.InitialContents.fill_(initial_contents)
            self.Contents.fill_(initial_contents)

        if el.FindElement("unusable-volume") is not None:
            self.UnusableVol.fill_(el.FindElementValueAsNumberConvertTo("unusable-volume", "GAL"))

        if el.FindElement("temperature") is not None:
            initial_temperature = el.FindElementValueAsNumber("temperature")
            self.InitialTemperature.fill_(initial_temperature)
            self.Temperature.fill_(initial_temperature)

        if el.FindElement("standpipe") is not None:
            initial_standpipe = el.FindElementValueAsNumberConvertTo("standpipe", "LBS")
            self.InitialStandpipe.fill_(initial_standpipe)
            self.Standpipe.fill_(initial_standpipe)

        if el.FindElement("priority") is not None:
            initial_priority = int(el.FindElementValueAsNumber("priority"))
            self.InitialPriority.fill_(initial_priority)
            self.Priority.fill_(initial_priority)

        if el.FindElement("density") is not None:
            self.Density.fill_(el.FindElementValueAsNumberConvertTo("density", "LBS/GAL"))

        fuelName = None
        if el.FindElement("type") is not None:
            fuelName = el.FindElementValue("type")

        self.SetPriority(self.InitialPriority)

        if torch.any(self.Capacity == 0.0):
            print("Tank capacity must not be zero. Reset to 0.00001 lbs!")
            self.Contents[self.Capacity == 0.0] = 0.0 #flipped order from original code so we don't have to save the mask
            self.Capacity[self.Capacity == 0.0] = 0.00001
        
        if torch.any(self.Capacity <= self.GetUnusable()):
            print("Tank capacity is lower than the amount of unusable fuel. Did you accidentally swap unusable and capacity?")
            raise Exception("Tank definition error")

        self.PctFull = 100.0 * self.Contents / self.Capacity

        # Check whether this is a solid propellant "tank". Initialize it if true.

        element_Grain = el.FindElement("grain_config")

        if element_Grain is not None:
            GType = element_Grain.GetAttributeValue("type")
            if GType == "CYLINDRICAL":
                grainType = GrainType.Cylindrical
            elif GType == "ENDBURNING":
                grainType = GrainType.EndBurning
            elif GType == "FUNCTION":
                print("Function grain not supported yet")
                raise Exception("Function grain not supported yet")
            else:
                print("Unknown grain type")
                raise Exception("Unknown grain type")

            if element_Grain.FindElement("length") is not None:
                self.Length.fill_(element_Grain.FindElementValueAsNumberConvertTo("length", "IN"))
            if element_Grain.FindElement("bore_diameter") is not None:
                self.InnerRadius.fill_(element_Grain.FindElementValueAsNumberConvertTo("bore_diameter", "IN") / 2.0)

            # Initialize solid propellant values for debug and runtime use
            if grainType == GrainType.Cylindrical:
                if torch.any(self.Radius <= self.InnerRadius):
                    print("The bore diameter should be smaller than the total grain diameter!")
                    raise Exception("Tank definition error")
                else:
                    self.Volume = torch.pi * self.Length * (self.Radius ** 2 - self.InnerRadius ** 2)  # cubic inches
            elif grainType == GrainType.EndBurning:
                self.Volume = torch.pi * self.Length * self.Radius ** 2 # cubic inches
            elif grainType == GrainType.Function:
                #irrelevant
                self.Volume = torch.ones(*self.size, 1.0, dtype=torch.float64, device=self.device)
            elif grainType == GrainType.Unknown:
                print("Unknown grain type found in this rocket engine definition.")
                raise Exception("Unknown grain type")
            
            self.Density = (self.Capacity * self.units.LB_TO_SLUG) / self.Volume # slugs/cubic inches #Note this is weird because density is used earlier.

        self.CalculateInertias()
    
        f_temp = self.FahrenheitToCelsius(self.Temperature[self.Temperature != -9999.0])
        self.InitialTemperature[self.Temperature != -9999.0] = f_temp.clone()
        self.Temperature[self.Temperature != -9999.0] = f_temp.clone()
        self.Area = 40.0 * torch.pow(self.Capacity / 1975, 0.666666667)
        if fuelName is not None and len(fuelName) > 0:
            self.Density.fill_(self.ProcessFuelName(fuelName))

        # TODO: Bind properties

    def GetDensity(self):
        return self.Density
    
    def GetType(self):
        return self.Type
    
    def ProcessFuelName(self, name: str) -> float:
        val = 0.0
        if name == "AVGAS":
            val = 6.02
        elif name == "JET-A":    
            val = 6.74
        elif name == "JET-A1":   
            val = 6.74
        elif name == "JET-B":    
            val = 6.48
        elif name == "JP-1":     
            val = 6.76
        elif name == "JP-2":     
            val = 6.38
        elif name == "JP-3":     
            val = 6.34
        elif name == "JP-4":     
            val = 6.48
        elif name == "JP-5":     
            val = 6.81
        elif name == "JP-6":     
            val = 6.55
        elif name == "JP-7":     
            val = 6.61
        elif name == "JP-8":     
            val = 6.66
        elif name == "JP-8+100": 
            val = 6.66
        #elif name == "JP-9":     val = 6.74
        #elif name == "JPTS":     val = 6.74
        elif name == "RP-1":     
            val = 6.73
        elif name == "T-1":      
            val = 6.88
        elif name == "ETHANOL":  
            val = 6.58
        elif name == "HYDRAZINE":
            val = 8.61
        elif name == "F-34":     
            val = 6.66
        elif name == "F-35":     
            val = 6.74
        elif name == "F-40":     
            val = 6.48
        elif name == "F-44":     
            val = 6.81
        elif name == "AVTAG":    
            val = 6.48
        elif name == "AVCAT":    
            val = 6.81
        else:
            print("Unknown fuel type")
            val = 6.6
        return val
    
    def FahrenheitToCelsius(self, temperature: torch.Tensor):
        return (temperature - 32.0) * 5.0 / 9.0
    
    def CalculateInertias(self):
        mass = self.Contents * self.units.LB_TO_SLUG
        rad2 = self.Radius ** 2

        if self.grainType != GrainType.Unknown:
            if torch.any(self.Density <= 0.0 and self.Contents > 0.0):
                print("Solid propellant grain density is zero!")
                raise Exception("Tank definition error")
            self.Volume[self.Density > 0.0] = self.Contents[self.Density > 0.0] * self.units.LB_TO_SLUG / self.Density[self.Density > 0.0]
            self.Volume[self.Contents <= 0.0] = 0.0

            if self.grainType == GrainType.Cylindrical:
                self.InnerRadius = torch.sqrt(rad2 - self.Volume / (torch.pi * self.Length))
                radsumsqr = (rad2 + self.InnerRadius ** 2) / 144.0
                self.Ixx = 0.5 * mass * radsumsqr
                self.Iyy = mass * (3.0 * radsumsqr + self.Length ** 2 / 144.0) / 12.0
                self.Izz = self.Iyy.clone()
            elif self.grainType == GrainType.EndBurning:
                self.Length = self.Volume / (torch.pi * rad2)

                self.Ixx = 0.5 * mass * rad2 / 144.0
                self.Iyy = mass * (3.0 * rad2 + self.Length ** 2) / (144.0 * 12.0)
                self.Izz = self.Iyy.clone()
            elif self.grainType == GrainType.Function:
                print("Function grain not supported yet")
                raise Exception("Function grain not supported yet")
            else:
                print("Unknown grain type")
                raise Exception("Unknown grain type")
        else: # assume liquid propellant: shrinking snowball
            pos_radius = self.Radius > 0.0
            val = mass[pos_radius] * self.InertiaFactor[pos_radius] * 0.4 * self.Radius[pos_radius]**2 / 144.0
            self.Ixx[pos_radius] = val.clone()
            self.Iyy[pos_radius] = val.clone()
            self.Izz[pos_radius] = val.clone()

    def GetUnusable(self) -> torch.Tensor:
        return self.UnusableVol * self.Density
    
    def SetPriority(self, priority: torch.Tensor):
        self.Priority = priority
        self.Selected = self.Priority > 0.0

    def GetXYZ(self) -> torch.Tensor:
        return self.vXYZ_drain + (self.Contents / self.Capacity) * (self.vXYZ - self.vXYZ_drain)
    
    def GetContents(self) -> torch.Tensor:
        return self.Contents

    def reset_to_ic(self):
        pass
    #SetTemperature( InitialTemperature );
    #SetStandpipe ( InitialStandpipe );
    #SetContents ( InitialContents );
    #PctFull = 100.0*Contents/Capacity;
    #SetPriority( InitialPriority );
    #CalculateInertias();
