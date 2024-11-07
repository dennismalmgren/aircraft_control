from typing import Optional
from enum import IntEnum
import torch
import os
from pathlib import Path

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.input_output.model_loader import ModelLoader
from jsbsim_parallel.input_output.model_path_provider import ModelPathProvider

class EulerAngles(IntEnum):
    Phi = 0
    Tht = 1
    Psi = 2
    
class ModelBase:
    def __init__(self, path_provider: ModelPathProvider,
                 *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        size = batch_size if batch_size is not None else torch.Size([])
        self.rate = torch.ones(*size, 1, dtype=torch.float64, device=device) #todo: probably a scalar
        self.path_provider = path_provider

    def Upload(self, el: Element, preload: bool) -> bool:
        loader = ModelLoader(self)
        document = loader.open(el)
        if not document:
            return False
        
        result = True
        if preload:
            result = self.LoadDocument(document)
       
        if document != el:
            el.MergeAttributes(document)
            #if preload:
                # TODO: Load local properties
        
            element = document.FindElement()
            while element is not None:
                el.AddChildElement(element)
                element.SetParent(el)
                element = document.FindNextElement()
        return result
    
    def LoadDocument(self, element: Element, prefix: str = "") -> bool:
        
        #TODO: Load local properties

        result = self.PreLoad(element, prefix)
        return result
    
    def PreLoad(self, element: Element, prefix: str = ""):
        # that's for ModelFunctions, not supported.    
        result = True
        func = element.FindElement("function")
        if func is not None:
            result = False
            print("Unsupported function found")
        return result
        
    def PostLoad(self, element: Element, prefix: str = "") -> bool:
        func = element.FindElement("function")
        result = True
        if func is not None:
            result = False
            print("Unsupported function found")
        return result
    

    def FindFullPathName(self, path: str):
        ap = self.path_provider.GetFullAircraftPath()
        return self.CheckPathName(ap, path)


    def CheckPathName(path: str, filename: Path) -> str:
        """
        Constructs a full path and verifies the file exists.
        Appends .xml if the file extension is missing.
        
        Args:
            path (Path): The directory path.
            filename (str): The file name.

        Returns:
            Path: Full path if the file exists, else an empty Path.
        """

        full_name = os.path.join(path, filename)
        filename, file_extension = os.path.splitext(full_name)
        if file_extension != ".xml":
            full_name = Path(full_name)
            full_name = path.with_suffix(".xml")
        return full_name if full_name.exists() else Path()

    def GetRate(self):
        return self.rate
    
    def init_model(self):
        pass
