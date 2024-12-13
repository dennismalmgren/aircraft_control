import jsbsim
import os
from dataclasses import dataclass
import numpy as np
from enum import Enum
from clib_interface.clib_exec import CLibExec, sim_state, sim_inputs, sim_initialconditions

@dataclass
class AircraftSimulatorConfig:
    sim_freq: int = 60

@dataclass
class SimulatorState:
    # TODO: Merge with api
    x: float
    y: float
    z: float
    
    v_aex: float
    v_aey: float
    v_aez: float

    v_ex: float
    v_ey: float
    v_ez: float

    a_ex: float
    a_ey: float
    a_ez: float

    my: float
    gamma: float
    chi: float
    alpha: float
    beta: float
    mach: float

    psi: float
    theta: float
    phi: float
    
    p: float
    q: float
    r: float

    pdot: float
    qdot: float
    rdot: float

    fuel: float

@dataclass
class AircraftCLibInitialConditions:
    x_rt90: float = 120.0 # geocentric longitude [deg]
    y_rt90: float = 60.0  # geodetic latitude  [deg]
    z_rt90: float = 6000      # altitude above mean sea level [ft]
    psi_deg: float = 0.0   # initial (true) heading [deg] (0, 360)
    u_mps: float = 800.0        # body frame x-axis velocity [ft/s]  (-2200, 2200)

class AircraftCLibSimulator:
    
    def __init__(self, config: AircraftSimulatorConfig):
        self.clib_exec = None
        self.sim_freq = config.sim_freq
        self.dt = 1.0 / self.sim_freq
        
    def reset(self, ic: AircraftCLibInitialConditions):
        self.clib_exec = CLibExec()

        self._set_initial_conditions(ic) #just assume it works, for now.

        state = self._get_state()
        return state
    
    def step(self, control_action: np.ndarray): 
        """
        control_action: np.ndarray = [roll, pitch, throttle]
        """
        roll = control_action[0]
        pitch = control_action[1]
        throttle = control_action[2]
        inputs = sim_inputs(
            roll = roll,
            pitch = pitch,
            throttle = throttle
        )
        self.clib_exec.set_controls(inputs)
        self.clib_exec.step()
        aircraft_state = self._get_state()
        return aircraft_state
    
    slugs_ft3_to_kg_m3 = 14.59390  / (0.3048**3)
    ft_to_m = 0.3048

    def _get_state(self):
        state: sim_state = self.clib_exec.get_simulator_state()

        simstate = SimulatorState(
            x = state.X,
            y = state.Y,
            z = state.Z,
            v_aex = state.V_AEX,
            v_aey = state.V_AEY,
            v_aez = state.V_AEX,
            v_ex = state.V_AEY,
            v_ey = state.V_AEX,
            v_ez = state.V_AEY,
            a_ex = state.V_AEX,
            a_ey = state.V_AEY,
            a_ez = state.V_AEX,
            my = state.V_AEY,
            gamma = state.V_AEX,
            chi = state.V_AEY,
            alpha = state.V_AEX,
            beta = state.V_AEY,
            mach = state.V_AEX,
            psi = state.PSI,
            theta = state.THETA,
            phi = state.PHI,
            p = state.V_AEY,
            q = state.V_AEX,
            r = state.V_AEY,
            pdot = state.V_AEX,
            qdot = state.V_AEY,
            rdot = state.V_AEX,
            fuel = state.V_AEY,
        )
        return simstate

    def _set_initial_conditions(self, ic: AircraftCLibInitialConditions):
        sim_init = sim_initialconditions(
            x_rt90=ic.x_rt90,
            y_rt90=ic.y_rt90,
            z_rt90=ic.z_rt90,
            psi_deg=ic.psi_deg,
            u_mps=ic.u_mps
        )
        self.clib_exec.set_initial_conditions(sim_init)

    def close(self):
        """ Closes the simulation and any plots. """
        if self.clib_exec:
            self.clib_exec = None