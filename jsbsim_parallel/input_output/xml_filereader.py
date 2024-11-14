import os
import xml.etree.ElementTree as ET
from typing import Optional
from jsbsim_parallel.input_output.element import Element

class XMLFileReader:
    def __init__(self):
        self.file_parser = None  # Placeholder for file parser, not directly used in Python.

    def load_xml_document(self, filename: str, verbose: bool = True) -> Optional[Element]:
        """
        Load and parse the XML document from the given filename.

        Args:
            filename (str): Path to the XML file.
            verbose (bool): Print errors if True.

        Returns:
            Optional[Element]: The root Element object if parsing is successful, None otherwise.
        """
        if not filename.endswith('.xml'):
            filename += '.xml'

        if not os.path.exists(filename):
            if verbose:
                print(f"Could not open file: {filename}")
            return None

        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            return self._parse_element(root)
        except ET.ParseError as e:
            if verbose:
                print(f"Error parsing XML file {filename}: {e}")
            return None

    def _parse_element(self, xml_elem: ET.Element) -> Element:
        """
        Recursively converts xml.etree.ElementTree.Element to custom Element.

        Args:
            xml_elem (ET.Element): The XML element to convert.

        Returns:
            Element: The converted Element instance.
        """
        elem = Element(xml_elem.tag)
        elem.file_name = xml_elem.base if hasattr(xml_elem, 'base') else ""
        elem.line_number = xml_elem.sourceline if hasattr(xml_elem, 'sourceline') else -1

        # Add attributes
        for key, value in xml_elem.attrib.items():
            elem.add_attribute(key, value)

        # Add data lines
        if xml_elem.text and xml_elem.text.strip():
            lines = xml_elem.text.split('\n')
            for line in lines:
                if len(line.strip()) > 0:
                    elem.add_data(line.strip())

        # Recursively parse child elements
        for child in xml_elem:
            child_elem = self._parse_element(child)
            elem.AddChildElement(child_elem)
            child_elem.SetParent(elem)
        return elem


# # Usage example
# if __name__ == "__main__":
#     reader = XMLFileReader()
#     root_element = reader.load_xml_document("example.xml", verbose=True)
#     if root_element:
#         print("Root element:", root_element)

