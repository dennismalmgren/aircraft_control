from typing import Optional
from enum import IntEnum

import torch

from jsbsim_parallel.input_output.element import Element, isfloat
from jsbsim_parallel.models.unit_conversions import UnitConversions
from jsbsim_parallel.models.propulsion_models.engine import Engine, EngineInputs, EngineType

from jsbsim_parallel.models.mass_balance import MassBalance
from jsbsim_parallel.input_output.simulator_service import SimulatorService
from jsbsim_parallel.math.function import Function
from jsbsim_parallel.math.parameter import Parameter
from jsbsim_parallel.math.real_value import RealValue


    
class PhaseType(IntEnum):
    Off = 0
    Run = 1
    SpinUp = 2
    Start = 3
    Stall = 4
    Seize = 5
    Trim = 6

class SimplifiedTSFC(Parameter):
    def __init__(self, _turb: 'Turbine', tsfcVal: float):
        self._turb = _turb
        self.tsfc = tsfcVal
    
    def GetName(self):
        return ""
    
    def GetValue(self) -> torch.Tensor:
        T = self._turb._in.Temperature
        N2norm = self._turb.N2norm
        return self.tsfc * torch.sqrt(T / 389.7) * (0.84 + (1 - N2norm)** 2)


class Turbine(Engine):
    def __init__(self, 
                 mass_balance: MassBalance,
                 simulator_service: SimulatorService,
                 el: Element, 
                 engine_number: int, 
                 inputs: EngineInputs,
                 *, device: torch.device = torch.device("cpu"), batch_size: Optional[torch.Size] = None):
        super().__init__(
            mass_balance,
            simulator_service,
            engine_number, 
            inputs, 
            device=device, batch_size=batch_size)
        # Define the size for batch support
        self.device = device
        self.size = batch_size if batch_size is not None else torch.Size([])
        self.Type = EngineType.Turbine

        self.MilThrust = torch.full((*self.size, 1), 10000.0, dtype=torch.float64, device=self.device)  # Maximum Unaugmented Thrust, static @ S.L. (lbf)
        self.MaxThrust = torch.full((*self.size, 1), 10000.0, dtype=torch.float64, device=self.device)  # Maximum Augmented Thrust, static @ S.L. (lbf)
    
        # TODO:
#        self.TSFC = #
    #    self.ATSFC = #
        self.IdleN1 = torch.full((*self.size, 1), 30.0, dtype=torch.float64, device=self.device)  # Idle N1
        self.IdleN2 = torch.full((*self.size, 1), 60.0, dtype=torch.float64, device=self.device)  # Idle N2

        self.MaxN1 = torch.full((*self.size, 1), 100.0, dtype=torch.float64, device=self.device)  # N1 at 100% throttle
        self.MaxN2 = torch.full((*self.size, 1), 100.0, dtype=torch.float64, device=self.device)  # N2 at 100% throttle

        self.Augmented = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device)  # = 1 if augmentation installed
        self.AugMethod = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device)  # see details for values
        self.Injected = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device)  # = 1 if water injection installed
        self.BypassRatio = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # Bypass Ratio
        self.BleedDemand = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        # std::shared_ptr<FGFunction> IdleThrustLookup;
        # std::shared_ptr<FGFunction> MilThrustLookup;
        # std::shared_ptr<FGFunction> MaxThrustLookup;
        # std::shared_ptr<FGFunction> InjectionLookup;

        self.N1_spinup = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)  # N1 spin up rate from pneumatic starter (per second)
        self.N2_spinup = torch.full((*self.size, 1), 3.0, dtype=torch.float64, device=self.device)  # N2 spin up rate from pneumatic starter (per second)
        self.IgnitionN1 = torch.full((*self.size, 1), 5.21, dtype=torch.float64, device=self.device)  # Ignition N1
        self.IgnitionN2 = torch.full((*self.size, 1), 25.18, dtype=torch.float64, device=self.device)  # Ignition N2
        self.N1_start_rate = torch.full((*self.size, 1), 1.4, dtype=torch.float64, device=self.device)  # N1 spin up rate from ignition (per second)
        self.N2_start_rate = torch.full((*self.size, 1), 2.0, dtype=torch.float64, device=self.device)  # N2 spin up rate from ignition (per second)
        self.N1_spindown = torch.full((*self.size, 1), 2.0, dtype=torch.float64, device=self.device)  # N1 spin down factor
        self.N2_spindown = torch.full((*self.size, 1), 2.0, dtype=torch.float64, device=self.device)  # N2 spin down factor

        self.InjectionTime = torch.full((*self.size, 1), 30.0, dtype=torch.float64, device=self.device)
        self.InjectionTimer = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.InjWaterNorm = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        self.EPR = torch.full((*self.size, 1), 1.0, dtype=torch.float64, device=self.device)
        self.disableWindmill = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)  # flag to disable windmilling of engine in Off phase

        self.Load(el)



        self.phase = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device) #to capture PhaseType enum



        # TODO
        self.N1 = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # N1
        self.N2 = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # N2
        self.N2norm = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # N2 normalized (0=idle, 1=max)


        
        self.IdleFF = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # Idle Fuel Flow (lbm/hr)

        self.N1_factor = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # factor to tie N1 and throttle
        self.N2_factor = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # factor to tie N2 and throttle
        self.ThrottlePos = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # FCS-supplied throttle position - modified for local use!
        self.AugmentCmd = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)  # modulated afterburner command (0.0 to 1.0)




        self.Stalled = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)  # true if engine is compressor-stalled
        self.Seized = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)  # true if inner spool is seized
        self.Overtemp = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)  # true if EGT exceeds limits
        self.Fire = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)  # true if engine fire detected
        self.Injection = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.Augmentation = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.Reversed = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)
        self.Cutoff = torch.zeros(*self.size, 1, dtype=torch.bool, device=self.device)

        self.Ignition = torch.zeros(*self.size, 1, dtype=torch.int32, device=self.device)


        self.EGT_degC = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.OilPressure_psi = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.OilTemp_degK = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.InletPosition = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.NozzlePosition = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.correctedTSFC = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        self.InjN1increment = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)
        self.InjN2increment = torch.zeros(*self.size, 1, dtype=torch.float64, device=self.device)

        # FGFDMExec *FDMExec;
        # std::shared_ptr<FGParameter> N1SpoolUp;
        # std::shared_ptr<FGParameter> N1SpoolDown;
        # std::shared_ptr<FGParameter> N2SpoolUp;
        # std::shared_ptr<FGParameter> N2SpoolDown;
    
    def Load(self, el: Element):
        function_element = el.FindElement("function")
        while function_element is not None:
            name = function_element.GetAttributeValue("name")
            if name == "IdleThrust" or name == "MilThrust" or name == "AugThrust" \
            or name == "Injection" or name == "N1SpoolUp" or name == "N1SpoolDown" \
            or name == "N2SpoolUp" or name == "N2SpoolDown":
                function_element.SetAttributeValue("name", "propulsion/engine[#]/" + name)
            function_element = el.FindNextElement("function")
        
        super().Load(el)
 
        # TODO: Reset to IC

        if el.FindElement("milthrust") is not None:
            self.MilThrust.fill_(el.FindElementValueAsNumberConvertTo("milthrust", "LBS"))

        if el.FindElement("maxthrust") is not None:
            self.MaxThrust.fill_(el.FindElementValueAsNumberConvertTo("maxthrust", "LBS"))

        if el.FindElement("bypassratio") is not None:
            self.BypassRatio.fill_(el.FindElementValueAsNumber("bypassratio"))

        if el.FindElement("bleed") is not None:
            self.BleedDemand.fill_(el.FindElementValueAsNumber("bleed"))

        if el.FindElement("ignitionn1") is not None:
            self.IgnitionN1.fill_(el.FindElementValueAsNumber("ignitionn1"))

        if el.FindElement("ignitionn2") is not None:
            self.IgnitionN2.fill_(el.FindElementValueAsNumber("ignitionn2"))

        if el.FindElement("idlen1") is not None:
            self.IdleN1.fill_(el.FindElementValueAsNumber("idlen1"))

        if el.FindElement("idlen2") is not None:
            self.IdleN2.fill_(el.FindElementValueAsNumber("idlen2"))

        if el.FindElement("maxn1") is not None:
            self.MaxN1.fill_(el.FindElementValueAsNumber("maxn1"))

        if el.FindElement("maxn2") is not None:
            self.MaxN2.fill_(el.FindElementValueAsNumber("maxn2"))

        if el.FindElement("n1spinup") is not None:
            self.N1_spinup.fill_(el.FindElementValueAsNumber("n1spinup"))

        if el.FindElement("n2spinup") is not None:
            self.N2_spinup.fill_(el.FindElementValueAsNumber("n2spinup"))

        if el.FindElement("n1startrate") is not None:
            self.N1_start_rate.fill_(el.FindElementValueAsNumber("n1startrate"))

        if el.FindElement("n2startrate") is not None:
            self.N2_start_rate.fill_(el.FindElementValueAsNumber("n2startrate"))

        if el.FindElement("n1spindown") is not None:
            self.N1_spindown.fill_(el.FindElementValueAsNumber("n1spindown"))

        if el.FindElement("n2spindown") is not None:
            self.N2_spindown.fill_(el.FindElementValueAsNumber("n2spindown"))

        if el.FindElement("augmented") is not None:
            self.Augmented.fill_(int(el.FindElementValueAsNumber("augmented")))

        if el.FindElement("augmethod") is not None:
            self.AugMethod.fill_(int(el.FindElementValueAsNumber("augmethod")))

        if el.FindElement("injected") is not None:
            self.Injected.fill_(int(el.FindElementValueAsNumber("injected")))

        if el.FindElement("injection-time") is not None:
            self.InjectionTime.fill_(el.FindElementValueAsNumber("injection-time"))
            self.InjWaterNorm.fill_(1.0)

        if el.FindElement("injection-N1-inc") is not None:
            self.InjN1increment.fill_(el.FindElementValueAsNumber("injection-N1-inc"))

        if el.FindElement("injection-N2-inc") is not None:
            self.InjN2increment.fill_(el.FindElementValueAsNumber("injection-N2-inc"))

        if el.FindElement("disable-windmill") is not None:
            self.disableWindmill.fill_(el.FindElementValueAsBoolean("disable-windmill"))

        property_prefix = f"propulsion/engine[{self.EngineNumber}]"
        self.IdleThrustLookup = self.GetPreFunction(property_prefix + "/IdleThrust")
        self.MilThrustLookup = self.GetPreFunction(property_prefix + "/MilThrust")
        self.MaxThrustLookup = self.GetPreFunction(property_prefix + "/AugThrust")
        self.InjectionLookup = self.GetPreFunction(property_prefix + "/Injection")

        tsfcElement = el.FindElement("tsfc")
        if tsfcElement:
            val = tsfcElement.GetDataLine()
            if isfloat(val.strip()):
                self.TSFC = SimplifiedTSFC(self, float(val))
            else:
                self.TSFC = Function(tsfcElement, str(self.EngineNumber), device=self.device, batch_size=self.size)


        atsfcElement = el.FindElement("atsfc")
        if atsfcElement:
            val = atsfcElement.GetDataLine()
            if isfloat(val.strip()):
                self.ATSFC = RealValue(float(val))
            else:
                self.ATSFC = Function(atsfcElement, str(self.EngineNumber), device=self.device, batch_size=self.size)

        self.N1_factor = self.MaxN1 - self.IdleN1
        self.N2_factor = self.MaxN2 - self.IdleN2
        self.OilTemp_degK = self._in.TAT_c + 273.0
        self.IdleFF = torch.pow(self.MilThrust, 0.2) * 107.0 # Just an estimate

        #TODO: bindmodel properties
        return True