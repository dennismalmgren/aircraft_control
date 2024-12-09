import ctypes
from dataclasses import dataclass

from .aircraft_simulator import AircraftCLibInitialConditions

class sim_state(ctypes.Structure):
    _fields_ = [
        ('X', ctypes.c_float),
        ('Y', ctypes.c_float),
        ('Z', ctypes.c_float),     
        ('V_AEX', ctypes.c_float),
        ('V_AEY', ctypes.c_float),
        ('V_AEZ', ctypes.c_float),       
        ('V_EX', ctypes.c_float),
        ('V_EY', ctypes.c_float),
        ('V_EZ', ctypes.c_float),
        ('MY', ctypes.c_float),
        ('GAMMA', ctypes.c_float),
        ('CHI', ctypes.c_float),  
        ('ALPHA', ctypes.c_float),
        ('BETA', ctypes.c_float),
        ('MACH', ctypes.c_float),  
        ('PSI', ctypes.c_float),
        ('THETA', ctypes.c_float),
        ('PHI', ctypes.c_float),  
        ('P', ctypes.c_float),
        ('Q', ctypes.c_float),
        ('R', ctypes.c_float),  
        ('PDOT', ctypes.c_float),
        ('QDOT', ctypes.c_float),
        ('RDOT', ctypes.c_float),
        ('FUEL', ctypes.c_float)
    ]

@dataclass
class sim_inputs:
    roll: float
    pitch: float
    throttle: float

class CLibExec:
    def __init__(self):
        self.my_sim = ctypes.CDLL("./libsiminterface.so")
        #set up methods
        #init: x, y, z, speed, heading
        self.my_sim.init_sim.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.my_sim.init_sim.restype = None

        #roll, pitch, throttle
        self.my_sim.set_inputs.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.my_sim.set_inputs.restype = None

        self.my_sim.get_state.argtypes = [ctypes.POINTER(sim_state)]
        self.my_sim.get_state.restype = None

        self.my_sim.step.argtypes = None
        self.my_sim.step.restype = None

    def set_initial_conditions(self, initial_conditions: AircraftCLibInitialConditions):
        # TODO: Check types (feet, deg etc)
        self.my_sim.init_sim(initial_conditions.x_rt90, initial_conditions.y_rt90, initial_conditions.z_rt90, initial_conditions.v_aex_mps, initial_conditions.psi_deg)
        

    def get_simulator_state(self) -> sim_state:
        the_state = sim_state()
        self.my_sim.get_state(the_state)

        return the_state

    def set_controls(self, inputs: sim_inputs):
        self.my_sim.set_inputs(inputs.roll, inputs.pitch, inputs.throttle)

    def step(self):
        self.my_sim.step()

    def close(self):
        self.my_sim = None


    