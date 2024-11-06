import math
from typing import List, Dict, Optional


class Element:
    conversion_factors = {
        # Length
        ("M", "FT"): 3.2808399,
        ("FT", "M"): 1 / 3.2808399,
        ("FT", "IN"): 12.0,
        ("IN", "FT"): 1 / 12.0,
        # Add more conversions here
    }

    def __init__(self, name: str):
        self.name = name
        self.attributes: Dict[str, str] = {}
        self.data_lines: List[str] = []
        self.children: List[Element] = []
        self.parent: Optional[Element] = None
        self.file_name: str = ""
        self.line_number: int = -1

    def has_attribute(self, key: str) -> bool:
        return key in self.attributes

    def GetAttributeValue(self, key: str) -> str:
        return self.attributes.get(key, "")

    def set_attribute_value(self, key: str, value: str) -> bool:
        self.attributes[key] = value
        return True

    def get_attribute_value_as_number(self, key: str) -> float:
        value = self.GetAttributeValue(key)
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Expected numeric attribute for '{key}', got '{value}'")

    def add_child(self, child: 'Element'):
        child.parent = self
        self.children.append(child)

    def get_num_children(self) -> int:
        return len(self.children)

    def get_data_as_number(self) -> float:
        if len(self.data_lines) != 1:
            raise ValueError("Expected single data line.")
        try:
            return float(self.data_lines[0])
        except ValueError:
            raise ValueError(f"Data line is not numeric: '{self.data_lines[0]}'")

    def FindElement(self, name: str) -> Optional['Element']:
        for child in self.children:
            if child.name == name:
                return child
        return None

    def add_attribute(self, key: str, value: str):
        self.attributes[key] = value

    def add_data(self, data: str):
        self.data_lines.append(data.strip())

    def convert_value(self, value: float, from_unit: str, to_unit: str) -> float:
        if from_unit == to_unit:
            return value
        factor = self.conversion_factors.get((from_unit, to_unit))
        if factor is None:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        return value * factor

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
