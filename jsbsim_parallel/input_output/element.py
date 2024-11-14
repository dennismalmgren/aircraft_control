import math
from typing import List, Dict, Optional
from collections import defaultdict
import os
import torch

def isfloat(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
    
class Element:
    conversions_initialized = False
    convert: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    def __init__(self, name: str):
        if not self.conversions_initialized:
            self.initialize_conversions()
            self.conversions_initialized = True

        self.name = name
        self.attributes: Dict[str, str] = {}
        self.data_lines: List[str] = []
        self.children: List[Element] = []
        self.parent: Optional[Element] = None
        self.file_name: str = ""
        self.line_number: int = -1
        self.element_index = 0

    def deep_convert_to_dict(self, d):
        """Recursively convert a nested defaultdict to a regular dict."""
        if isinstance(d, defaultdict):
            d = {k: self.deep_convert_to_dict(v) for k, v in d.items()}
        return d

    def initialize_conversions(self):
        self.convert["M"]["FT"] = 3.2808399
        self.convert["FT"]["M"] = 1.0/self.convert["M"]["FT"]
        self.convert["CM"]["FT"] = 0.032808399
        self.convert["FT"]["CM"] = 1.0/self.convert["CM"]["FT"]
        self.convert["KM"]["FT"] = 3280.8399
        self.convert["FT"]["KM"] = 1.0/self.convert["KM"]["FT"]
        self.convert["FT"]["IN"] = 12.0
        self.convert["IN"]["FT"] = 1.0/self.convert["FT"]["IN"]
        self.convert["IN"]["M"] = self.convert["IN"]["FT"] * self.convert["FT"]["M"]
        self.convert["M"]["IN"] = self.convert["M"]["FT"] * self.convert["FT"]["IN"]
        # Area
        self.convert["M2"]["FT2"] = self.convert["M"]["FT"]*self.convert["M"]["FT"]
        self.convert["FT2"]["M2"] = 1.0/self.convert["M2"]["FT2"]
        self.convert["CM2"]["FT2"] = self.convert["CM"]["FT"]*self.convert["CM"]["FT"]
        self.convert["FT2"]["CM2"] = 1.0/self.convert["CM2"]["FT2"]
        self.convert["M2"]["IN2"] = self.convert["M"]["IN"]*self.convert["M"]["IN"]
        self.convert["IN2"]["M2"] = 1.0/self.convert["M2"]["IN2"]
        self.convert["FT2"]["IN2"] = 144.0
        self.convert["IN2"]["FT2"] = 1.0/self.convert["FT2"]["IN2"]
        # Volume
        self.convert["IN3"]["CC"] = 16.387064
        self.convert["CC"]["IN3"] = 1.0/self.convert["IN3"]["CC"]
        self.convert["FT3"]["IN3"] = 1728.0
        self.convert["IN3"]["FT3"] = 1.0/self.convert["FT3"]["IN3"]
        self.convert["M3"]["FT3"] = 35.3146667
        self.convert["FT3"]["M3"] = 1.0/self.convert["M3"]["FT3"]
        self.convert["LTR"]["IN3"] = 61.0237441
        self.convert["IN3"]["LTR"] = 1.0/self.convert["LTR"]["IN3"]
        self.convert["GAL"]["FT3"] = 0.133681
        self.convert["FT3"]["GAL"] = 1.0/self.convert["GAL"]["FT3"]
        self.convert["IN3"]["GAL"] = self.convert["IN3"]["FT3"]*self.convert["FT3"]["GAL"]
        self.convert["LTR"]["GAL"] = self.convert["LTR"]["IN3"]*self.convert["IN3"]["GAL"]
        self.convert["M3"]["GAL"] = 1000.*self.convert["LTR"]["GAL"]
        self.convert["CC"]["GAL"] = self.convert["CC"]["IN3"]*self.convert["IN3"]["GAL"]
        # Mass & Weight
        self.convert["LBS"]["KG"] = 0.45359237
        self.convert["KG"]["LBS"] = 1.0/self.convert["LBS"]["KG"]
        self.convert["SLUG"]["KG"] = 14.59390
        self.convert["KG"]["SLUG"] = 1.0/self.convert["SLUG"]["KG"]
        # Moments of Inertia
        self.convert["SLUG*FT2"]["KG*M2"] = 1.35594
        self.convert["KG*M2"]["SLUG*FT2"] = 1.0/self.convert["SLUG*FT2"]["KG*M2"]
        # Angles
        self.convert["RAD"]["DEG"] = 180.0/math.pi
        self.convert["DEG"]["RAD"] = 1.0/self.convert["RAD"]["DEG"]
        # Angular rates
        self.convert["RAD/SEC"]["DEG/SEC"] = self.convert["RAD"]["DEG"]
        self.convert["DEG/SEC"]["RAD/SEC"] = 1.0/self.convert["RAD/SEC"]["DEG/SEC"]
        # Spring force
        self.convert["LBS/FT"]["N/M"] = 14.5939
        self.convert["N/M"]["LBS/FT"] = 1.0/self.convert["LBS/FT"]["N/M"]
        # Damping force
        self.convert["LBS/FT/SEC"]["N/M/SEC"] = 14.5939
        self.convert["N/M/SEC"]["LBS/FT/SEC"] = 1.0/self.convert["LBS/FT/SEC"]["N/M/SEC"]
        # Damping force (Square Law)
        self.convert["LBS/FT2/SEC2"]["N/M2/SEC2"] = 47.880259
        self.convert["N/M2/SEC2"]["LBS/FT2/SEC2"] = 1.0/self.convert["LBS/FT2/SEC2"]["N/M2/SEC2"]
        # Power
        self.convert["WATTS"]["HP"] = 0.001341022
        self.convert["HP"]["WATTS"] = 1.0/self.convert["WATTS"]["HP"]
        # Force
        self.convert["N"]["LBS"] = 0.22482
        self.convert["LBS"]["N"] = 1.0/self.convert["N"]["LBS"]
        # Velocity
        self.convert["KTS"]["FT/SEC"] = 1.6878098571
        self.convert["FT/SEC"]["KTS"] = 1.0/self.convert["KTS"]["FT/SEC"]
        self.convert["M/S"]["FT/S"] = 3.2808399
        self.convert["M/S"]["KTS"] = self.convert["M/S"]["FT/S"]/self.convert["KTS"]["FT/SEC"]
        self.convert["M/SEC"]["FT/SEC"] = 3.2808399
        self.convert["FT/S"]["M/S"] = 1.0/self.convert["M/S"]["FT/S"]
        self.convert["M/SEC"]["FT/SEC"] = 3.2808399
        self.convert["FT/SEC"]["M/SEC"] = 1.0/self.convert["M/SEC"]["FT/SEC"]
        self.convert["KM/SEC"]["FT/SEC"] = 3280.8399
        self.convert["FT/SEC"]["KM/SEC"] = 1.0/self.convert["KM/SEC"]["FT/SEC"]
        # Torque
        self.convert["FT*LBS"]["N*M"] = 1.35581795
        self.convert["N*M"]["FT*LBS"] = 1/self.convert["FT*LBS"]["N*M"]
        # Valve
        self.convert["M4*SEC/KG"]["FT4*SEC/SLUG"] = self.convert["M"]["FT"]*self.convert["M"]["FT"]*\
        self.convert["M"]["FT"]*self.convert["M"]["FT"]/self.convert["KG"]["SLUG"]
        self.convert["FT4*SEC/SLUG"]["M4*SEC/KG"] = \
        1.0/self.convert["M4*SEC/KG"]["FT4*SEC/SLUG"]
        # Pressure
        self.convert["INHG"]["PSF"] = 70.7180803
        self.convert["PSF"]["INHG"] = 1.0/self.convert["INHG"]["PSF"]
        self.convert["ATM"]["INHG"] = 29.9246899
        self.convert["INHG"]["ATM"] = 1.0/self.convert["ATM"]["INHG"]
        self.convert["PSI"]["INHG"] = 2.03625437
        self.convert["INHG"]["PSI"] = 1.0/self.convert["PSI"]["INHG"]
        self.convert["INHG"]["PA"] = 3386.0 # inches Mercury to pascals
        self.convert["PA"]["INHG"] = 1.0/self.convert["INHG"]["PA"]
        self.convert["LBS/FT2"]["N/M2"] = 14.5939/self.convert["FT"]["M"]
        self.convert["N/M2"]["LBS/FT2"] = 1.0/self.convert["LBS/FT2"]["N/M2"]
        self.convert["LBS/FT2"]["PA"] = self.convert["LBS/FT2"]["N/M2"]
        self.convert["PA"]["LBS/FT2"] = 1.0/self.convert["LBS/FT2"]["PA"]
        # Mass flow
        self.convert["KG/MIN"]["LBS/MIN"] = self.convert["KG"]["LBS"]
        self.convert["KG/SEC"]["LBS/SEC"] = self.convert["KG"]["LBS"]
        self.convert["N/SEC"]["LBS/SEC"] = 0.224808943
        self.convert["LBS/SEC"]["N/SEC"] = 1.0/self.convert["N/SEC"]["LBS/SEC"]
        # Fuel Consumption
        self.convert["LBS/HP*HR"]["KG/KW*HR"] = 0.6083
        self.convert["KG/KW*HR"]["LBS/HP*HR"] = 1.0/self.convert["LBS/HP*HR"]["KG/KW*HR"]
        # Density
        self.convert["KG/L"]["LBS/GAL"] = 8.3454045
        self.convert["LBS/GAL"]["KG/L"] = 1.0/self.convert["KG/L"]["LBS/GAL"]
        # Gravitational
        self.convert["FT3/SEC2"]["M3/SEC2"] = self.convert["FT3"]["M3"]
        self.convert["M3/SEC2"]["FT3/SEC2"] = self.convert["M3"]["FT3"]

        # Length
        self.convert["M"]["M"] = 1.00
        self.convert["KM"]["KM"] = 1.00
        self.convert["FT"]["FT"] = 1.00
        self.convert["IN"]["IN"] = 1.00
        # Area
        self.convert["M2"]["M2"] = 1.00
        self.convert["FT2"]["FT2"] = 1.00
        # Volume
        self.convert["IN3"]["IN3"] = 1.00
        self.convert["CC"]["CC"] = 1.0
        self.convert["M3"]["M3"] = 1.0
        self.convert["FT3"]["FT3"] = 1.0
        self.convert["LTR"]["LTR"] = 1.0
        self.convert["GAL"]["GAL"] = 1.0
        # Mass & Weight
        self.convert["KG"]["KG"] = 1.00
        self.convert["LBS"]["LBS"] = 1.00
        # Moments of Inertia
        self.convert["KG*M2"]["KG*M2"] = 1.00
        self.convert["SLUG*FT2"]["SLUG*FT2"] = 1.00
        # Angles
        self.convert["DEG"]["DEG"] = 1.00
        self.convert["RAD"]["RAD"] = 1.00
        # Angular rates
        self.convert["DEG/SEC"]["DEG/SEC"] = 1.00
        self.convert["RAD/SEC"]["RAD/SEC"] = 1.00
        # Spring force
        self.convert["LBS/FT"]["LBS/FT"] = 1.00
        self.convert["N/M"]["N/M"] = 1.00
        # Damping force
        self.convert["LBS/FT/SEC"]["LBS/FT/SEC"] = 1.00
        self.convert["N/M/SEC"]["N/M/SEC"] = 1.00
        # Damping force (Square law)
        self.convert["LBS/FT2/SEC2"]["LBS/FT2/SEC2"] = 1.00
        self.convert["N/M2/SEC2"]["N/M2/SEC2"] = 1.00
        # Power
        self.convert["HP"]["HP"] = 1.00
        self.convert["WATTS"]["WATTS"] = 1.00
        # Force
        self.convert["N"]["N"] = 1.00
        # Velocity
        self.convert["FT/SEC"]["FT/SEC"] = 1.00
        self.convert["KTS"]["KTS"] = 1.00
        self.convert["M/S"]["M/S"] = 1.0
        self.convert["M/SEC"]["M/SEC"] = 1.0
        self.convert["KM/SEC"]["KM/SEC"] = 1.0
        # Torque
        self.convert["FT*LBS"]["FT*LBS"] = 1.00
        self.convert["N*M"]["N*M"] = 1.00
        # Valve
        self.convert["M4*SEC/KG"]["M4*SEC/KG"] = 1.0
        self.convert["FT4*SEC/SLUG"]["FT4*SEC/SLUG"] = 1.0
        # Pressure
        self.convert["PSI"]["PSI"] = 1.00
        self.convert["PSF"]["PSF"] = 1.00
        self.convert["INHG"]["INHG"] = 1.00
        self.convert["ATM"]["ATM"] = 1.0
        self.convert["PA"]["PA"] = 1.0
        self.convert["N/M2"]["N/M2"] = 1.00
        self.convert["LBS/FT2"]["LBS/FT2"] = 1.00
        # Mass flow
        self.convert["LBS/SEC"]["LBS/SEC"] = 1.00
        self.convert["KG/MIN"]["KG/MIN"] = 1.0
        self.convert["LBS/MIN"]["LBS/MIN"] = 1.0
        self.convert["N/SEC"]["N/SEC"] = 1.0
        # Fuel Consumption
        self.convert["LBS/HP*HR"]["LBS/HP*HR"] = 1.0
        self.convert["KG/KW*HR"]["KG/KW*HR"] = 1.0
        # Density
        self.convert["KG/L"]["KG/L"] = 1.0
        self.convert["LBS/GAL"]["LBS/GAL"] = 1.0
        # Gravitational
        self.convert["FT3/SEC2"]["FT3/SEC2"] = 1.0
        self.convert["M3/SEC2"]["M3/SEC2"] = 1.0
        # Electrical
        self.convert["VOLTS"]["VOLTS"] = 1.0
        self.convert["OHMS"]["OHMS"] = 1.0
        self.convert["AMPERES"]["AMPERES"] = 1.0
        self.convert = self.deep_convert_to_dict(self.convert)

    def GetElement(self, el: int = 0) -> 'Element':
        if len(self.children) > el:
            self.element_index = el
            return self.children[el]
        else:
            self.element_index = 0
            return None

    def GetNumDataLines(self):
        return len(self.data_lines)
    
    def GetNumElements(self, element_name: str = None) -> int:
        if not element_name:
            return len(self.children)
        else:
            number_of_elements = 0
            el = self.FindElement(element_name)
            while el:
                number_of_elements += 1
                el = self.FindNextElement(element_name)
            return number_of_elements


    def GetNextElement(self) -> 'Element':
        if len(self.children) > self.element_index + 1:
            self.element_index += 1
            return self.children[self.element_index]
        else:
            self.element_index = 0
            return None
        
    def HasAttribute(self, key: str) -> bool:
        return key in self.attributes

    def GetAttributeValue(self, key: str) -> str:
        return self.attributes.get(key, "")

    def SetAttributeValue(self, key: str, value: str) -> bool:
        self.attributes[key] = value
        return True

    def GetAttributeValueAsNumber(self, key: str) -> float:
        value = self.GetAttributeValue(key)
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Expected numeric attribute for '{key}', got '{value}'")

    def AddChildElement(self, child: 'Element'):
        self.children.append(child)

    def SetParent(self, element: 'Element'):
        self.parent = element   

    def GetParent(self):
        return self.parent

    def get_num_children(self) -> int:
        return len(self.children)
    
    #TODO: Take output result as input
    def FindElementTripletConvertTo(self, target_units: str, *, device:torch.device, batch_size: Optional[torch.Size]) -> torch.Tensor:
        batch_size = batch_size if batch_size else torch.Size([])

        triplet = torch.zeros(*batch_size, 3, dtype=torch.float64, device=device) 
        
        # Get the supplied units from the attributes
        supplied_units = self.GetAttributeValue("unit")

        # Check for unit conversion availability
        if supplied_units:
            if supplied_units not in self.convert:
                raise ValueError(f"Supplied unit: '{supplied_units}' does not exist.")
            if target_units not in self.convert[supplied_units]:
                raise ValueError(f"Supplied unit: '{supplied_units}' cannot be converted to {target_units}.")

        # Define triplet axis names for x, y, z with alternative names roll, pitch, yaw
        axes = [("x", "roll"), ("y", "pitch"), ("z", "yaw")]
        for i, (primary, alternative) in enumerate(axes):
            item = self.FindElement(primary) or self.FindElement(alternative)
            if item:
                value = item.GetDataAsNumber()
                # Apply conversion if supplied units are specified
                if supplied_units:
                    value *= self.convert[supplied_units][target_units]
                # Disperse value if needed
                triplet[..., i] = self.DisperseValue(item, value, supplied_units, target_units)
            else:
                triplet[..., i] = 0.0  # Set default if item is not found

        return triplet

    def FindElementValue(self, el: str) -> str:
        element = self.FindElement(el)
        if element is not None:
            return element.GetDataLine()
        else:
            return ""
    
    def GetDataLine(self, i: int = 0):
        if len(self.data_lines) > 0:
            return self.data_lines[i]
        else:
            return ""
        
    def FindElementValueAsBoolean(self, el: str) -> bool:
        element = self.FindElement(el)
        if element is None:
            print(f"Attempting to get non-existent element {el}")
            return False
        
        value = element.GetDataAsNumber()
        if value == 0:
            return False
        else:
            return True

    def FindElementValueAsNumber(self, el: str) -> float:
        element = self.FindElement(el)

        if element is None:
           raise ValueError(f"Attempting to get non-existent element {el}")

        value = element.GetDataAsNumber()
        value = self.DisperseValue(element, value)
        return value
    
    def FindElementValueAsNumberConvertTo(self, el: str, target_units: str) -> float:
        element = self.FindElement(el)

        if element is None:
           raise ValueError(f"Attempting to get non-existent element {el}")

        supplied_units = element.GetAttributeValue("unit")
        if supplied_units:
            if supplied_units not in self.convert:
                raise ValueError(f"Supplied unit: \"{supplied_units}\" does not exist (typo?).")

            if target_units not in self.convert[supplied_units]:
                raise ValueError(f"Supplied unit: \"{supplied_units}\" cannot be converted to {target_units}")
        
        # Get the numeric value from the element
        value = element.GetDataAsNumber()

        # Sanity checks for angular values
        if supplied_units == "RAD" and abs(value) > 2 * math.pi:
            print(f"{element.GetName()} value {value} RAD is outside the range [-2π RAD, +2π RAD]")

        if supplied_units == "DEG" and abs(value) > 360.0:
            print(f"{element.GetName()} value {value} DEG is outside the range [-360 DEG, +360 DEG]")

        # Convert the value if units are supplied
        if supplied_units:
            value *= self.convert[supplied_units][target_units]

        # Final sanity checks after conversion
        if target_units == "RAD" and abs(value) > 2 * math.pi:
            print(f"{element.GetName()} value {value} RAD is outside the range [-2π RAD, +2π RAD]")
       
        if target_units == "DEG" and abs(value) > 360.0:
            print(f"{element.GetName()} value {value} DEG is outside the range [-360 DEG, +360 DEG]")

        # Call DisperseValue if it's defined elsewhere
        value = self.DisperseValue(element, value, supplied_units, target_units)

        return value
    
    def DisperseValue(self, element: 'Element', value: float, supplied_units: str = "", target_units: str = "") -> float:
        # Check the environment variable
        disperse = os.getenv("JSBSIM_DISPERSE") == "1"  
        if element.HasAttribute("dispersion") and disperse:
            print('Not supported')
        return value
    
    def GetName(self) -> str:
        return self.name
    
    def GetDataAsNumber(self) -> float:
        if len(self.data_lines) != 1:
            raise ValueError("Expected single data line.")
        try:
            return float(self.data_lines[0])
        except ValueError:
            raise ValueError(f"Data line is not numeric: '{self.data_lines[0]}'")

    def FindNextElement(self, el: str = "") -> Optional['Element']:
        """Find the next element with the specified name or return the next child if no name is specified."""
        if len(el) == 0:  # If no name is specified, return the next child
            if self.element_index < len(self.children):
                element = self.children[self.element_index]
                self.element_index += 1
                return element
            else:
                self.element_index = 0
                return None

        # Look for the next element with the specified name
        for i in range(self.element_index, len(self.children)):
            if self.children[i].GetName() == el:
                self.element_index = i + 1
                return self.children[i]

        # Reset index if no matching element is found
        self.element_index = 0
        return None

    def FindElement(self, el: str = "") -> Optional['Element']:
        if len(el) == 0 and len(self.children) >= 1:
            self.element_index = 1
            return self.children[0]

        for i, child in enumerate(self.children):
            if el == child.GetName():
                self.element_index = i + 1
                return child

        self.element_index = 0
        return None

    def add_attribute(self, key: str, value: str):
        self.attributes[key] = value

    def add_data(self, data: str):
        self.data_lines.append(data.strip())

    def MergeAttributes(self, element: 'Element'):
        for key, value in element.attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value
            # else:
            #     # If debug level is > 0 and values differ, print debug message
            #     if self.debug_lvl > 0 and self.attributes[key] != value:
            #         print(
            #             f"{el.ReadFrom()} Attribute '{key}' is overridden in file "
            #             f"{self.GetFileName()}: line {self.GetLineNumber()}\n"
            #             f"The value '{self.attributes[key]}' will be used instead of '{value}'."
            #         )
    # def convert_value(self, value: float, from_unit: str, to_unit: str) -> float:
    #     if from_unit == to_unit:
    #         return value
    #     factor = self.convert.get((from_unit, to_unit))
    #     if factor is None:
    #         raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    #     return value * factor

    def __repr__(self) -> str:
        return f"Element(name={self.name}, attributes={self.attributes}, data_lines={self.data_lines})"


# # Usage example:
# element = Element("LengthConversion")
# element.add_attribute("unit", "M")
# element.add_data("10.0")

# # Convert value from meters to feet
# meters = element.get_data_as_number()
# feet = element.convert_value(meters, "M", "FT")
# print(f"{meters} meters is {feet} feet")
