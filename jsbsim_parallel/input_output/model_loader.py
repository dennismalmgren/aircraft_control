from typing import Optional
import os
from pathlib import Path

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.input_output.xml_filereader import XMLFileReader
from jsbsim_parallel.input_output.simulator_service import SimulatorService

class ModelLoader:
    def __init__(self, model: SimulatorService):
        self.model = model  # Assuming model is an instance with a FindFullPathName method.
        self.cached_files = {}

    def open(self, el: Element) -> Optional[Element]:
        """
        Loads an XML document if the element has a 'file' attribute.
        If the file is already loaded, returns the cached document.
        
        Args:
            el (Element): The base XML Element containing a potential file reference.
        
        Returns:
            Optional[Element]: The loaded or cached Element, or None if loading failed.
        """
        document = el
        fname = el.GetAttributeValue("file")

        if fname:
            xml_reader = XMLFileReader()
            path = Path(fname)

            # Resolve relative paths using the model's method, if available
            if not path.is_absolute():
                path = self.model.FindFullPathName(str(path))

            path_str = str(path)
            if path_str in self.cached_files:
                document = self.cached_files[path_str]
            else:
                document = xml_reader.load_xml_document(path_str)
                if document is None:
                    print(f"Error: Could not open file: {fname}")
                    return None
                self.cached_files[path_str] = document

            if document.GetName() != el.GetName():
                document.SetParent(el)
                el.AddChildElement(document)

        return document