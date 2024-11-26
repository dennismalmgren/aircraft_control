import jsbsim
import os
from dataclasses import dataclass
from .catalog import JSBSimCatalog
import numpy as np
from enum import Enum

@dataclass
class AircraftSimulatorConfig:
    sim_freq: int = 60
    jsbsim_module_dir: str = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')
    aircraft_model: str = 'f16'


@dataclass
class SimulatorState:
    velocity_u_m_sec: float
    velocity_v_m_sec: float
    velocity_w_m_sec: float
    velocity_north_m_sec: float
    velocity_east_m_sec: float
    velocity_down_m_sec: float
    velocity_mach: float
    acceleration_udot_m_sec2: float
    acceleration_vdot_m_sec2: float
    acceleration_wdot_m_sec2: float
    attitude_psi_rad: float
    attitude_theta_rad: float
    attitude_phi_rad: float
    velocity_p_rad_sec: float
    velocity_q_rad_sec: float
    velocity_r_rad_sec: float
    acceleration_pdot_rad_sec2: float
    acceleration_qdot_rad_sec2: float
    acceleration_rdot_rad_sec2: float
    position_lat_geod_rad: float
    position_long_gc_rad: float
    position_h_sl_m: float
    rho_kg_m3: float
    a_m_sec: float
    crosswind_m_sec: float
    headwind_m_sec: float
    vc_m_sec: float
    vg_m_sec: float

@dataclass
class AircraftJSBSimInitialConditions:
    long_gc_deg: float = 120.0 # geocentric longitude [deg]
    lat_geod_deg: float = 60.0  # geodetic latitude  [deg]
    h_sl_ft: float = 6000      # altitude above mean sea level [ft]
    psi_true_deg: float = 0.0   # initial (true) heading [deg] (0, 360)
    u_fps: float = 800.0        # body frame x-axis velocity [ft/s]  (-2200, 2200)
    v_fps: float = 0.0          # body frame y-axis velocity [ft/s]  (-2200, 2200)
    w_fps: float = 0.0          # body frame z-axis velocity [ft/s]  (-2200, 2200)
    p_rad_sec: float = 0.0      # roll rate  [rad/s]  (-2 * pi, 2 * pi)
    q_rad_sec: float = 0.0      # pitch rate [rad/s]  (-2 * pi, 2 * pi)
    r_rad_sec: float = 0.0      # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
    roc_fpm: float = 0.0        # initial rate of climb [ft/min]
    terrain_elevation_ft: float = 0.0     # terrain elevation [ft] 
    
class AircraftJSBSimSimulator:
    state_properties = [
        #linear velocity
        JSBSimCatalog.velocities_u_fps,
        JSBSimCatalog.velocities_v_fps,
        JSBSimCatalog.velocities_w_fps,
        #NED velocity
        JSBSimCatalog.velocities_v_north_fps,
        JSBSimCatalog.velocities_v_east_fps,
        JSBSimCatalog.velocities_v_down_fps,
        #linear acceleration
        JSBSimCatalog.accelerations_udot_ft_sec2,
        JSBSimCatalog.accelerations_vdot_ft_sec2,
        JSBSimCatalog.accelerations_wdot_ft_sec2,
        #attitude 
        JSBSimCatalog.attitude_phi_rad,
        JSBSimCatalog.attitude_theta_rad,
        JSBSimCatalog.attitude_psi_rad,
        #angular velocity
        JSBSimCatalog.velocities_p_rad_sec,
        JSBSimCatalog.velocities_q_rad_sec,
        JSBSimCatalog.velocities_r_rad_sec,
        #angular acceleration
        JSBSimCatalog.accelerations_pdot_rad_sec2,
        JSBSimCatalog.accelerations_qdot_rad_sec2,
        JSBSimCatalog.accelerations_rdot_rad_sec2,
        #navigation
        JSBSimCatalog.position_lat_geod_rad,
        JSBSimCatalog.position_long_gc_rad,
        JSBSimCatalog.position_h_sl_ft,
        #JSBSimCatalog.velocities_h_dot_fps,
        #environmental
        JSBSimCatalog.atmosphere_rho_slugs_ft3,
        JSBSimCatalog.atmosphere_a_fps,
        JSBSimCatalog.atmosphere_crosswind_fps,
        JSBSimCatalog.atmosphere_headwind_fps,
        #performance
        JSBSimCatalog.velocities_vc_fps,
        JSBSimCatalog.velocities_vg_fps,
        #speed
        JSBSimCatalog.velocities_mach
    ]

    def __init__(self, config: AircraftSimulatorConfig):
        self.jsbsim_exec = None
        self.sim_freq = config.sim_freq
        self.jsbsim_module_dir = config.jsbsim_module_dir
        self.aircraft_model = config.aircraft_model 
        self.dt = 1.0 / self.sim_freq
        self.jsbsim_catalog = JSBSimCatalog()
        
    def reset(self, ic: AircraftJSBSimInitialConditions):
        self.jsbsim_exec = jsbsim.FGFDMExec(self.jsbsim_module_dir)
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.aircraft_model)
        self.jsbsim_exec.set_dt(self.dt)

        #TODO: 
        #jsbsim_props = self.jsbsim_exec.query_property_catalog("").split('\n')
        #prop_names = [jsbsim_props[i].split()[0] for i in range(len(jsbsim_props)) if len(jsbsim_props[i]) > 0]
                #we could potentially pull jsbsim-props into catalog if we wanted.
                #maybe one could set a dictionary of the attributes they want fetched.
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
        control_action: np.ndarray = [throttle, aileron, elevator, rudder]
        """
        aileron = control_action[0]
        elevator = control_action[1]
        rudder = control_action[2]
        throttle = control_action[3]
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_throttle_cmd_norm.name_jsbsim, throttle)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_aileron_cmd_norm.name_jsbsim, aileron)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_elevator_cmd_norm.name_jsbsim, elevator)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.fcs_rudder_cmd_norm.name_jsbsim, rudder)
        
        result = self.jsbsim_exec.run()
        if not result:
            raise RuntimeError("JSBSim failed.")
        #todo: update properties (and return them)
        aircraft_state = self._get_state()
        return aircraft_state
    
    slugs_ft3_to_kg_m3 = 14.59390  / (0.3048**3)
    ft_to_m = 0.3048

    def _get_property(self, property, unit_conversion=1.0):
        return self.jsbsim_exec.get_property_value(property.name_jsbsim) * unit_conversion
    
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

    def _set_initial_conditions(self, ic: AircraftJSBSimInitialConditions):
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_long_gc_deg.name_jsbsim, ic.long_gc_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_lat_geod_deg.name_jsbsim, ic.lat_geod_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_h_sl_ft.name_jsbsim, ic.h_sl_ft)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_psi_true_deg.name_jsbsim, ic.psi_true_deg)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_u_fps.name_jsbsim, ic.u_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_v_fps.name_jsbsim, ic.v_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_w_fps.name_jsbsim, ic.w_fps)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_p_rad_sec.name_jsbsim, ic.p_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_q_rad_sec.name_jsbsim, ic.q_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_r_rad_sec.name_jsbsim, ic.r_rad_sec)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_roc_fpm.name_jsbsim, ic.roc_fpm)
        self.jsbsim_exec.set_property_value(self.jsbsim_catalog.ic_terrain_elevation_ft.name_jsbsim, ic.terrain_elevation_ft)

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim_exec:
            self.jsbsim_exec = None