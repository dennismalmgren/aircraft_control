from typing import Optional, Dict, List
from enum import IntEnum

import torch

#from jsbsim_parallel.models.fcs import FCS
from jsbsim_parallel.input_output.element import Element
from jsbsim_parallel.math.property_value import PropertyValue
from jsbsim_parallel.math.real_value import RealValue
from jsbsim_parallel.math.parameter_value import ParameterValue
from jsbsim_parallel.input_output.property_manager import PropertyNode

class FCSComponent:
    def __init__(self, 
                 fcs,
                 element: Element,
                 *, 
                 device: torch.device, batch_size: Optional[torch.Size] = None):
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.delay = 0.0 #tensor?
        self.fcs = fcs
        self.Input = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.Output = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.delay_time = torch.zeros(*self.size, 1, dtype=torch.float64, device=device)
        self.ClipMin = RealValue(0.0)
        self.ClipMax = RealValue(0.0)
        self.clip = False
        self.cyclic_clip = False
        self.dt = fcs.GetChannelDeltaT()
        self.InitNodes: List[PropertyValue] = []
        self.InputNodes: List[PropertyValue] = []
        self.OutputNodes: List[PropertyNode] = []
        self.output_array: List[torch.Tensor] = []

        if element.GetName() == "lag_filter":
            self.Type = "LAG_FILTER"
        elif element.GetName() == "lead_lag_filter":
            self.Type = "LEAD_LAG_FILTER"
        elif element.GetName() == "washout_filter":
            self.Type = "WASHOUT_FILTER"
        elif element.GetName() == "second_order_filter":
            self.Type = "SECOND_ORDER_FILTER"
        elif element.GetName() == "integrator":
            self.Type = "INTEGRATOR"
        elif element.GetName() == "summer":
            self.Type = "SUMMER"
        elif element.GetName() == "pure_gain":
            self.Type = "PURE_GAIN"
        elif element.GetName() == "scheduled_gain":
            self.Type = "SCHEDULED_GAIN"
        elif element.GetName() == "aerosurface_scale":
            self.Type = "AEROSURFACE_SCALE"
        elif element.GetName() == "switch":
            self.Type = "SWITCH"
        elif element.GetName() == "kinematic":
            self.Type = "KINEMATIC"
        elif element.GetName() == "deadband":
            self.Type = "DEADBAND"
        elif element.GetName() == "fcs_function":
            self.Type = "FCS_FUNCTION"
        elif element.GetName() == "pid":
            self.Type = "PID"
        elif element.GetName() == "sensor":
            self.Type = "SENSOR"
        elif element.GetName() == "accelerometer":
            self.Type = "ACCELEROMETER"
        elif element.GetName() == "magnetometer":
            self.Type = "MAGNETOMETER"
        elif element.GetName() == "gyro":
            self.Type = "GYRO"
        elif element.GetName() == "actuator":
            self.Type = "ACTUATOR"
        elif element.GetName() == "waypoint_heading":
            self.Type = "WAYPOINT_HEADING"
        elif element.GetName() == "waypoint_distance":
            self.Type = "WAYPOINT_DISTANCE"
        elif element.GetName() == "angle":
            self.Type = "ANGLE"
        elif element.GetName() == "distributor":
            self.Type = "DISTRIBUTOR"
        else:
            self.Type = "UNKNOWN"

        self.Name = element.GetAttributeValue("name")

        init_element = element.FindElement("init")
        while init_element:
            self.InitNodes.append(PropertyValue(init_element.GetDataLine(), init_element, 
                                                device=self.device, batch_size=self.size))
            init_element = element.FindNextElement("init")

        input_element = element.FindElement("input")
        while input_element:
            input = PropertyValue(input_element.GetDataLine(), input_element,
                                        device=self.device, batch_size=self.size)
            self.InputNodes.append(input)
            input_element = element.FindNextElement("input")
        
        #     # TODO
        # out_elem = element.FindElement("output")
        # while out_elem:
        #     output_node_name = out_elem.GetDataLine()
        #     node_exists = propertyManager.HasNode(...)
        #     outputNode = propertyManager.GetNode(...)
        #     if not outputNode:
        #         raise Exception("Output node not found")
        #     self.OutputNodes.append(outputNode)
        #     if not node_exists:
        #         outputNode.setDoubleValue(self.Output)
        #     out_elem = element.FindNextElement("output")

        delay_elem = element.FindElement("delay")
        if delay_elem:
            delay_str = delay_elem.GetDataLine()
            delayParam = ParameterValue(delay_str, delay_elem, 
                                        device=self.device, batch_size=self.size)
            delay_time = delayParam.GetValue()
            delayType = delay_elem.GetAttributeValue("type")
            if len(delayType) > 0:
                if delayType == "time":
                    self.delay = int(delay_time / self.dt)
                elif delayType == "frames":
                    self.delay = int(delay_time)
                else:
                    raise Exception("Unallowed delay type")
            else:
                self.delay = int(delay_time / self.dt)
            self.output_array = [torch.zeros(self.size, 1, dtype=torch.float64, device=self.device)] * self.delay


        clip_el = element.FindElement("clipto")
        if clip_el:
            el = clip_el.FindElement("min")
            if not el:
                print("Element <min> is missing, <clipto> is ignored")
        
            self.ClipMin = ParameterValue(None, el, device=self.device, batch_size=self.size)

            el = clip_el.FindElement("max")
            if not el:
                print("Element <max> is missing, <clipto> is ignored")
                self.ClipMin = None #TODO: Probably a jsbsim bug

            self.ClipMax = ParameterValue(None, el, device=self.device, batch_size=self.size)

            if clip_el.GetAttributeValue("type") == "cyclic":
                self.cyclic_clip = True
            
            self.clip = True


    def CheckInputNodes(self, MinNodes: int, MaxNodes: int, el: Element):
        num = len(self.InputNodes)
        if num < MinNodes:
            raise Exception("Not enough input nodes")
        
        if num > MaxNodes:
            print("Too many input nodes. The last input nodes will be ignored.")