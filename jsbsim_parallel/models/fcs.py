from typing import Optional, List, Union
from enum import IntEnum

import torch

from jsbsim_parallel.models.lgear import BrakeGroup
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.models.model_base import ModelBase
from jsbsim_parallel.input_output.property_manager import PropertyNode
from jsbsim_parallel.models.fcs_channel import FCSChannel
from jsbsim_parallel.flight_control.filter import Filter
from jsbsim_parallel.flight_control.gain import Gain
from jsbsim_parallel.flight_control.kinemat import Kinemat
from jsbsim_parallel.flight_control.switch import Switch
from jsbsim_parallel.flight_control.summer import Summer
from jsbsim_parallel.flight_control.pid import PID
from jsbsim_parallel.flight_control.fcs_function import FCSFunction
from jsbsim_parallel.input_output.simulator_service import SimulatorService
class OutputForm(IntEnum):
    Rad = 0
    Deg = 1
    Norm = 2
    Mag = 3
    NForms = 4


class SystemType(IntEnum):
    FCS = 0
    System = 1
    AutoPilot = 2

class FCS(ModelBase):
    def __init__(self,
                  simulator_service: SimulatorService,
                 *,
                 device: torch.device, batch_size: Optional[torch.Size] = None):
        #channelrate 1
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.simulator_service = simulator_service
        self.DaCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DeCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DrCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DfCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DsbCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.DspCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.PTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.YTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.RTrimCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.GearCmd = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        self.GearPos = torch.ones(*self.size, 1, dtype=torch.float64, device=device)
        #vectors
        self.ThrottleCmd: List[torch.Tensor] = []
        self.ThrottlePos = []
        self.MixtureCmd = []
        self.MixturePos = []
        self.PropAdvanceCmd: List[torch.Tensor] = []
        self.PropAdvance: List[torch.Tensor] = []
        self.PropFeatherCmd: List[torch.Tensor] = []
        self.PropFeather: List[torch.Tensor] = []
        self.BrakePos: List[torch.Tensor] = [torch.zeros(*self.size, 1, dtype=torch.float64, device=device) for _ in range(BrakeGroup.NumBrakeGroups)]
        #brakepos resize..
        self.TailhookPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.WingFoldPos = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)

        #bind
        self.DePos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DaLPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DaRPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DrPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DfPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DsbPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        self.DspPos = torch.zeros(*self.size, OutputForm.NForms, dtype=torch.float64, device=device)
        
        self.SystemChannels: List[FCSChannel] = []

    def GetPropFeather(self) -> List[torch.Tensor]:
        return self.PropFeather
    
    def GetPropAdvance(self) -> List[torch.Tensor]:
        return self.PropAdvance
    
    def GetMixtureCmd(self) -> List[torch.Tensor]:
        return self.MixtureCmd
    
    def GetMixturePos(self) -> List[torch.Tensor]:
        return self.MixturePos

    def AddThrottle(self):
        self.ThrottleCmd.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.ThrottlePos.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.MixtureCmd.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.MixturePos.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.PropAdvanceCmd.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.PropAdvance.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.PropFeatherCmd.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        self.PropFeather.append(torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device))
        
        #TODO: bind properties

    def GetThrottleCmd(self) -> List[torch.Tensor]:
        return self.ThrottleCmd
        
    def GetThrottlePos(self, engine: Optional[int] = None) -> Union[List[torch.Tensor], torch.Tensor]:
        if engine is None:
            return self.ThrottlePos
        else:
            return self.ThrottlePos[engine]

    def GetBrakePos(self) -> List[torch.Tensor]:
        return self.BrakePos
    
    def GetGearPos(self) -> torch.Tensor:
        return self.GearPos
    
    def Load(self, document: Element) -> bool:
        if document.GetName() == "autopilot":
            raise Exception("autopilot is not supported")
        elif document.GetName() == "flight_control":
            self.Name = "FCS"
            self.systype = SystemType.FCS
        elif document.GetName() == "system":
            raise Exception("system is not supported")

        if not super().Upload(document, True):
            return False
        
        self.Name += document.GetAttributeValue("name")

        channel_element = document.FindElement("channel")
        while channel_element:
            sOnOffProperty = channel_element.GetAttributeValue("execute")
            sChannelName = channel_element.GetAttributeValue("name")

            if len(channel_element.GetAttributeValue("execrate")) > 0:
                self.ChannelRate = channel_element.GetAttributeValueAsNumber("execrate")
            else:
                self.ChannelRate = 1

            if len(sOnOffProperty) > 0:
                raise Exception("On/Off is not supported")
            else:
                newChannel = FCSChannel(self, sChannelName, self.ChannelRate, device=self.device, batch_size=self.size)
            self.SystemChannels.append(newChannel)

            component_element = channel_element.GetElement()
            while component_element:
                if component_element.GetName() in ["lag_filter", "lead_lag_filter",
                                                   "washout_filter", "second_order_filter"]:
                    newChannel.Add(Filter(self, component_element))
                elif component_element.GetName() in ["pure_gain", "scheduled_gain",
                                                   "aerosurface_scale"]:
                    newChannel.Add(Gain(self, component_element, device=self.device, batch_size=self.size))
                elif component_element.GetName() in ["summer"]:
                    newChannel.Add(Summer(self, component_element, device=self.device, batch_size=self.size))
                elif component_element.GetName() in ["switch"]:
                    newChannel.Add(Switch(self, component_element, device=self.device, batch_size=self.size))
                elif component_element.GetName() in ["kinematic"]:
                    newChannel.Add(Kinemat(self, component_element, device=self.device, batch_size=self.size))
                elif component_element.GetName() in ["fcs_function"]:
                    newChannel.Add(FCSFunction(self, component_element, device=self.device, batch_size=self.size))
                elif component_element.GetName() in ["pid"]:
                    newChannel.Add(PID(self, component_element, device=self.device, batch_size=self.size))
                component_element = channel_element.GetNextElement()
            channel_element = document.FindNextElement("channel")
        return True
    
    def GetChannelDeltaT(self):
        return self.GetDt() * self.ChannelRate
    
    def GetDt(self):
        return self.simulator_service.GetDeltaT()

    def run(self, holding: bool) -> bool:
        if holding:
            return True

        return False

    def init_model(self) -> bool:

        #reset to IC.
        return True

