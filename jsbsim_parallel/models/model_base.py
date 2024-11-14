from typing import Optional, List
from enum import IntEnum
import torch
import os
from pathlib import Path

from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.input_output.model_loader import ModelLoader
from jsbsim_parallel.input_output.simulator_service import SimulatorService
from jsbsim_parallel.math.function import Function

class EulerAngles(IntEnum):
    Phi = 0
    Tht = 1
    Psi = 2
    
class ModelBase:
    def __init__(self, simulator_service: SimulatorService,
                 *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        self.rate = torch.ones(*self.size, 1, dtype=torch.float64, device=device) #todo: probably a scalar
        self.simulator_service = simulator_service
        self.PreFunctions: List[Function] = []
        self.PostFunctions: List[Function] = []

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
        while func is not None:
            fType = func.GetAttributeValue("type")
            if len(fType) == 0 or fType == "pre":
                mkfunc = Function(func, prefix, device=self.device, batch_size=self.size)
                self.PreFunctions.append(mkfunc)
            elif fType == "template":
                print("Unsupported function type (template) found") 
                result = False
                break
            func = element.FindNextElement("function")

        return result
        
    def PostLoad(self, element: Element, prefix: str = "") -> bool:
        result = True
        func = element.FindElement("function")
        while func is not None:
            fType = func.GetAttributeValue("type")
            if fType == "post":
                mkfunc = Function(func, prefix, device=self.device, batch_size=self.size)
                self.PostFunctions.append(mkfunc)
            func = element.FindNextElement("function")
        return result
    
    def GetPreFunction(self, name: str) -> Function:
        for f in self.PreFunctions:
            if f.GetName() == name:
                return f
        return None
    
    def FindFullPathName(self, path: str):
        ap = self.simulator_service.GetFullAircraftPath()
        return self.CheckPathName(ap, path)


    def CheckPathName(self, path: str, filename: Path) -> str:
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
            full_name = full_name.with_suffix(".xml")
        return str(full_name) if full_name.exists() else ""

    def GetRate(self):
        return self.rate
    
    def init_model(self):
        pass
