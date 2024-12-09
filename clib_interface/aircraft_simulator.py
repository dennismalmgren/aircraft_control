import jsbsim
import os
from dataclasses import dataclass
import numpy as np
from enum import Enum
from .catalog import CLibCatalog

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
        self.clib_exec = jsbsim.FGFDMExec(self.jsbsim_module_dir)
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.aircraft_model)
        self.jsbsim_exec.set_dt(self.dt)

        self._set_initial_conditions(ic)

        success = self.jsbsim_exec.run_ic()
        if not success:
            raise ValueError("Failed to run initial conditions in JSBSim.")
        
        # propulsion init running
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()
        state = self._get_state()
        return state
    
    def step(self, control_action: np.ndarray): 
        """
        control_action: np.ndarray = [roll, pitch, throttle]
        """
        roll = control_action[0]
        pitch = control_action[1]
        throttle = control_action[2]
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_throttle_cmd_norm.name, throttle)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_aileron_cmd_norm.name, aileron)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_elevator_cmd_norm.name, elevator)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_rudder_cmd_norm.name, rudder)
        
        result = self.jsbsim_exec.run()
        if not result:
            raise RuntimeError("JSBSim failed.")
        #todo: update properties (and return them)
        aircraft_state = self._get_state()
        return aircraft_state
    
    slugs_ft3_to_kg_m3 = 14.59390  / (0.3048**3)
    ft_to_m = 0.3048

    def _get_property(self, property, unit_conversion=1.0):
        return self.jsbsim_exec.get_property_value(property.name) * unit_conversion
    
    def _get_state(self):
        simstate = SimulatorState(
            velocity_u_m_sec = self._get_property(self.jsbsim_catalog.velocities_u_fps, self.ft_to_m),
            velocity_v_m_sec = self._get_property(self.jsbsim_catalog.velocities_v_fps, self.ft_to_m),
            velocity_w_m_sec = self._get_property(self.jsbsim_catalog.velocities_w_fps, self.ft_to_m),
            velocity_north_m_sec = self._get_property(self.jsbsim_catalog.velocities_v_north_fps, self.ft_to_m),
            velocity_east_m_sec = self._get_property(self.jsbsim_catalog.velocities_v_east_fps, self.ft_to_m),
            velocity_down_m_sec = self._get_property(self.jsbsim_catalog.velocities_v_down_fps, self.ft_to_m),            
            acceleration_udot_m_sec2 =  self._get_property(self.jsbsim_catalog.accelerations_udot_ft_sec2, self.ft_to_m),
            acceleration_vdot_m_sec2 = self._get_property(self.jsbsim_catalog.accelerations_vdot_ft_sec2, self.ft_to_m),
            acceleration_wdot_m_sec2 =  self._get_property(self.jsbsim_catalog.accelerations_wdot_ft_sec2, self.ft_to_m),
            attitude_psi_rad = self._get_property(self.jsbsim_catalog.attitude_psi_rad),
            attitude_theta_rad = self._get_property(self.jsbsim_catalog.attitude_theta_rad),
            attitude_phi_rad = self._get_property(self.jsbsim_catalog.attitude_phi_rad),
            velocity_p_rad_sec = self._get_property(self.jsbsim_catalog.velocities_p_rad_sec),
            velocity_q_rad_sec = self._get_property(self.jsbsim_catalog.velocities_q_rad_sec),
            velocity_r_rad_sec = self._get_property(self.jsbsim_catalog.velocities_r_rad_sec),
            acceleration_pdot_rad_sec2 = self._get_property(self.jsbsim_catalog.accelerations_pdot_rad_sec2),
            acceleration_qdot_rad_sec2 = self._get_property(self.jsbsim_catalog.accelerations_qdot_rad_sec2),
            acceleration_rdot_rad_sec2 = self._get_property(self.jsbsim_catalog.accelerations_rdot_rad_sec2),
            position_lat_geod_rad = self._get_property(self.jsbsim_catalog.position_lat_geod_rad),
            position_long_gc_rad = self._get_property(self.jsbsim_catalog.position_long_gc_rad),
            position_h_sl_m = self._get_property(self.jsbsim_catalog.position_h_sl_ft, self.ft_to_m),
            rho_kg_m3 = self._get_property(self.jsbsim_catalog.atmosphere_rho_slugs_ft3, self.slugs_ft3_to_kg_m3),
            a_m_sec = self._get_property(self.jsbsim_catalog.atmosphere_a_fps, self.ft_to_m),
            crosswind_m_sec = self._get_property(self.jsbsim_catalog.atmosphere_crosswind_fps, self.ft_to_m),
            headwind_m_sec = self._get_property(self.jsbsim_catalog.atmosphere_headwind_fps, self.ft_to_m),
            vc_m_sec = self._get_property(self.jsbsim_catalog.velocities_vc_fps, self.ft_to_m),
            vg_m_sec = self._get_property(self.jsbsim_catalog.velocities_vg_fps, self.ft_to_m),
            velocity_mach = self._get_property(self.jsbsim_catalog.velocities_mach)
        )
        return simstate

    def _set_initial_conditions(self, ic: AircraftCLibInitialConditions):
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_long_gc_deg.name, ic.long_gc_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_lat_geod_deg.name, ic.lat_geod_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_h_sl_ft.name, ic.h_sl_ft)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_psi_true_deg.name, ic.psi_true_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_u_fps.name, ic.u_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_v_fps.name, ic.v_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_w_fps.name, ic.w_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_p_rad_sec.name, ic.p_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_q_rad_sec.name, ic.q_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_r_rad_sec.name, ic.r_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_roc_fpm.name, ic.roc_fpm)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_terrain_elevation_ft.name, ic.terrain_elevation_ft)

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None