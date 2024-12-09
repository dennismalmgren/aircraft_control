from typing import Union, Optional, List
from dataclasses import dataclass
import os
import math

import torch
from torchrl.envs import EnvBase
from gymnasium import spaces
import numpy as np
from torchrl.data.tensor_specs import (
    #Binary,
    Bounded,
    Composite,
    Categorical,
    #DiscreteTensorSpec,
    #MultiOneHotDiscreteTensorSpec,
    #MultiDiscreteTensorSpec,
    #OneHotDiscreteTensorSpec,
    #TensorSpec,
    Unbounded,
)
from tensordict import TensorDict, TensorDictBase

from jsbsim_interface import AircraftJSBSimSimulator, AircraftSimulatorConfig, AircraftJSBSimInitialConditions, SimulatorState
from curriculum.curriculum_manager_jsbsim import CurriculumManagerJsbSim


DeviceType = Union[torch.device, str, int]

@dataclass
class JSBSimControlEnvConfig:
    agent_uids: List[str]
    agent_teams: List[str]

class JSBSimControlEnv(EnvBase):
    def __init__(self,
                 *,
                 device: DeviceType = None,
                 batch_size: Optional[torch.Size] = None,
                 run_type_checks: bool = False,
                 allow_done_after_reset: bool = False):
        super().__init__(device=device, batch_size=batch_size, run_type_checks=run_type_checks, allow_done_after_reset=allow_done_after_reset)
        self.uid = "A0100"

        config = AircraftSimulatorConfig(jsbsim_module_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../py_modules/JSBSim'))
        self.aircraft_simulator = AircraftJSBSimSimulator(config)

        self.observation_spec = Composite(
            u = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            v = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            w = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            mach = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            udot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            vdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            wdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            phi = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            theta = Unbounded(shape=(1,), device=device, dtype=torch.float32), 
            psi = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            p = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            q = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            r = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            pdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            qdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            rdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            lat = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            lon = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            alt = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            v_north = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            v_east = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            v_down = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            air_density = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            speed_of_sound = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            crosswind = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            headwind = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            true_airspeed = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            groundspeed = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            last_action = Unbounded(shape=(4,), device=device, dtype=torch.float32),
            #goals
            target_alt = Unbounded(shape=(1,), device=device, dtype=torch.float32), 
            target_speed = Unbounded(shape=(1,), device=device, dtype=torch.float32), 
            target_heading = Unbounded(shape=(1,), device=device, dtype=torch.float32), 
        )

        #note that throttle is 0->1 in sim.
        #aileron, elevator, rudder, throttle
        self.action_spec = Bounded(low = torch.tensor([-1.0, -1.0, -1.0, 0.0]),
                                   high = torch.tensor([1.0, 1.0, 1.0, 2.0]),
                                   device=device,
                                   dtype=torch.float32)
        
        self.done_spec = Composite(
                            terminated=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                            truncated=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                            done=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                        )
        
        self.reward_spec = Composite(
            reward = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_task = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_smoothness = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_safety = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_alt = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_speed = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_heading = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_roll = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_smoothness_pdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_smoothness_qdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            reward_smoothness_rdot = Unbounded(shape=(1,), device=device, dtype=torch.float32),
        )
        self.state_spec = self.observation_spec.clone()

        self._target_altitude = None
        self._target_speed = None
        self._target_heading = None
        self._tolerance_altitude = None
        self._tolerance_speed = None
        self._tolerance_heading = None
        self._tolerance_roll = None
        self._error_scale_alt = None
        self._error_scale_speed = None
        self._error_scale_heading = None
        self._error_scale_roll = None

        self.curriculum_manager = CurriculumManagerJsbSim(
            min_lat_geod_deg = 57.0,
            max_lat_geod_deg = 60.0,
            min_long_gc_deg = 15.0,
            max_long_gc_deg = 20.0,
            min_altitude = 1000.0,
            max_altitude = 10000.0,
            min_speed = 100.0,
            max_speed = 365.0,
            min_heading = 0.0,
            max_heading = 360.0
        )

    def heading_error(self, theta1, theta2):
        """
        Calculate the shortest heading error between two angles (in radians).
        Handles batch computations with PyTorch tensors.

        Parameters:
        - theta1: Current heading (tensor)
        - theta2: Desired heading (tensor)

        Returns:
        - error: Signed heading error in radians (tensor)
        """
        sin_diff = torch.sin(theta2) * torch.cos(theta1) - torch.cos(theta2) * torch.sin(theta1)
        cos_diff = torch.cos(theta2) * torch.cos(theta1) + torch.sin(theta2) * torch.sin(theta1)
        error = torch.atan2(sin_diff, cos_diff)
        return error

    def _add_reward(self, simulator_state: SimulatorState, td_out):
        safety_reward = torch.zeros(1, dtype=torch.float32, device=self.device)
        if simulator_state.position_h_sl_m < 300:
            safety_reward = torch.tensor(-100.0, dtype=torch.float32, device=self.device)
        else:
            safety_reward = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        alt_reward = 0.0
        speed_reward = 0.0
        heading_reward = 0.0
        roll_reward = 0.0
        total_reward = 0.0
        smoothness_reward = 0.0
        task_reward = torch.zeros(1, dtype=torch.float32, device=self.device)

        alt_error = torch.tensor(simulator_state.position_h_sl_m - self._target_altitude, dtype=torch.float32, device=self.device)

        if self._tolerance_altitude is not None:
            alt_error = torch.max(0, torch.abs(alt_error) - self._tolerance_altitude)
        alt_reward = torch.exp(-((alt_error / self._error_scale_alt) ** 2))
        
        speed_error = simulator_state.velocity_mach - self._target_speed

        if self._tolerance_speed is not None:
            speed_error = torch.max(0, torch.abs(speed_error) - self._tolerance_speed)

        speed_reward = torch.exp(-((speed_error / self._error_scale_speed)**2))
        
        current_heading = torch.tensor(simulator_state.attitude_psi_rad, dtype=torch.float32, device=self.device)
        heading_error = self.heading_error(current_heading, self._target_heading)
        
        if self._tolerance_heading is not None:
            heading_error = torch.max(0, torch.abs(heading_error) - self._tolerance_heading)

        heading_reward = torch.exp(-((heading_error / self._error_scale_heading)**2))
        
        roll_error = torch.tensor(simulator_state.attitude_phi_rad, dtype=torch.float32, device=self.device)

        if self._tolerance_roll is not None:
            roll_error = torch.max(0, abs(roll_error) - self._tolerance_roll)

        roll_reward = torch.exp(-((roll_error / self._error_scale_roll)**2))

        smoothness_pdot_error = torch.tensor(simulator_state.acceleration_pdot_rad_sec2, dtype=torch.float32, device=self.device)
        smoothness_pdot_reward = torch.exp(-((smoothness_pdot_error / self._error_scale_smoothness_pdot)**2))

        smoothness_qdot_error = torch.tensor(simulator_state.acceleration_qdot_rad_sec2, dtype=torch.float32, device=self.device)
        smoothness_qdot_reward = torch.exp(-((smoothness_qdot_error / self._error_scale_smoothness_qdot)**2))

        smoothness_rdot_error = torch.tensor(simulator_state.acceleration_rdot_rad_sec2, dtype=torch.float32, device=self.device)
        smoothness_rdot_reward = torch.exp(-((smoothness_rdot_error / self._error_scale_smoothness_rdot)**2))
            
        smoothness_reward = math.pow(smoothness_pdot_reward * smoothness_qdot_reward * smoothness_rdot_reward , 1/3)
        task_reward = torch.pow(alt_reward * speed_reward * heading_reward * roll_reward, 1/4)

        td_out.set("reward_safety", safety_reward)
        td_out.set("reward_alt", alt_reward)
        td_out.set("reward_speed", speed_reward)
        td_out.set("reward_heading", heading_reward)
        td_out.set("reward_roll", roll_reward)
        td_out.set("reward_smoothness_pdot", smoothness_pdot_reward)
        td_out.set("reward_smoothness_qdot", smoothness_qdot_reward)
        td_out.set("reward_smoothness_rdot", smoothness_rdot_reward)
        
        #TODO: Add errors as observations.
        td_out.set("reward_smoothness", smoothness_reward)
        td_out.set("reward_task", task_reward)
        
        #Combine thresholded rewards
        if not torch.isclose(safety_reward, torch.zeros(1, dtype=torch.float32, device=self.device)):
            total_reward = safety_reward
        else:
            total_reward = task_reward #no smoothness for now
        td_out.set("reward", total_reward)
        
    def _evaluate_terminated(self, simulator_state: SimulatorState) -> bool:
        if simulator_state.position_h_sl_m < 300:
            return True
        else:
            return False
    
    def _add_done_flags(self, simulator_state: SimulatorState, tensordict: TensorDict):
        terminated = torch.tensor([self._evaluate_terminated(simulator_state)], device=self.device)
        truncated = torch.tensor([False], device=self.device)
        tensordict.set("terminated", terminated)
        tensordict.set("truncated", truncated)
        tensordict.set("done", terminated or truncated)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        if tensordict is not None:
            td_out = tensordict.clone().empty()
        else:
            td_out = TensorDict({}, batch_size=self.batch_size, device=self.device)
        last_action = tensordict["action"].detach()
        action_np = last_action.cpu().numpy()
        simulator_state = self.aircraft_simulator.step(action_np)
        self._add_reward(simulator_state, td_out)
        self._add_observations(simulator_state, td_out)
        self._add_last_action(last_action, td_out)
        self._add_done_flags(simulator_state, td_out)
        return td_out


    def _add_observations(self, simulator_state: SimulatorState, tensordict: TensorDict):
        tensordict["u"] =  torch.tensor([simulator_state.velocity_u_m_sec], device=self.device)
        tensordict["v"] =  torch.tensor([simulator_state.velocity_v_m_sec], device=self.device)
        tensordict["w"] =  torch.tensor([simulator_state.velocity_w_m_sec], device=self.device)
        tensordict["mach"] = torch.tensor([simulator_state.velocity_mach], device=self.device)
        tensordict["v_north"] =  torch.tensor([simulator_state.velocity_north_m_sec], device=self.device)
        tensordict["v_east"] =  torch.tensor([simulator_state.velocity_east_m_sec], device=self.device)                
        tensordict["v_down"] =  torch.tensor([simulator_state.velocity_down_m_sec], device=self.device)
        tensordict["udot"] =  torch.tensor([simulator_state.acceleration_udot_m_sec2], device=self.device)
        tensordict["vdot"] =  torch.tensor([simulator_state.acceleration_vdot_m_sec2], device=self.device)
        tensordict["wdot"] =  torch.tensor([simulator_state.acceleration_wdot_m_sec2], device=self.device)
        tensordict["phi"] =  torch.tensor([simulator_state.attitude_phi_rad], device=self.device)
        tensordict["theta"] =  torch.tensor([simulator_state.attitude_theta_rad], device=self.device)
        tensordict["psi"] =  torch.tensor([simulator_state.attitude_psi_rad], device=self.device)
        tensordict["p"] =  torch.tensor([simulator_state.velocity_p_rad_sec], device=self.device)
        tensordict["q"] =  torch.tensor([simulator_state.velocity_q_rad_sec], device=self.device)
        tensordict["r"] =  torch.tensor([simulator_state.velocity_r_rad_sec], device=self.device)
        tensordict["pdot"] =  torch.tensor([simulator_state.acceleration_pdot_rad_sec2], device=self.device)
        tensordict["qdot"] =  torch.tensor([simulator_state.acceleration_qdot_rad_sec2], device=self.device)
        tensordict["rdot"] =  torch.tensor([simulator_state.acceleration_rdot_rad_sec2], device=self.device)
        tensordict["lat"] =  torch.tensor([simulator_state.position_lat_geod_rad], device=self.device)
        tensordict["lon"] =  torch.tensor([simulator_state.position_long_gc_rad], device=self.device)
        tensordict["alt"] =  torch.tensor([simulator_state.position_h_sl_m], device=self.device)
        tensordict["air_density"] =  torch.tensor([simulator_state.rho_kg_m3], device=self.device)
        tensordict["speed_of_sound"] =  torch.tensor([simulator_state.a_m_sec], device=self.device)
        tensordict["crosswind"] =  torch.tensor([simulator_state.crosswind_m_sec], device=self.device)
        tensordict["headwind"] =  torch.tensor([simulator_state.headwind_m_sec], device=self.device)
        tensordict["true_airspeed"] =  torch.tensor([simulator_state.vc_m_sec], device=self.device)
        tensordict["groundspeed"] =  torch.tensor([simulator_state.vg_m_sec], device=self.device)
        tensordict["target_alt"] =  torch.tensor([self._target_altitude], device=self.device)
        tensordict["target_speed"] =  torch.tensor([self._target_speed], device=self.device)
        tensordict["target_heading"] =  torch.tensor([self._target_heading], device=self.device)
        return tensordict

    def _add_last_action(self, action: torch.tensor, tensordict: TensorDict):
        tensordict["last_action"] = action

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            td_out = tensordict.clone().empty()
        else:
            td_out = TensorDict({}, batch_size=self.batch_size, device=self.device)
        if "aircraft_ic" in kwargs:
            aircraft_ic = kwargs["aircraft_ic"]
        else:
            aircraft_ic = self.curriculum_manager.get_initial_conditions()
        primer_action = torch.zeros((self.action_spec.shape[-1],), device=self.device)
        simulator_state = self.aircraft_simulator.reset(aircraft_ic)
        self._target_altitude = torch.tensor(simulator_state.position_h_sl_m, dtype=torch.float32, device=self.device)
        self._tolerance_altitude = None #100
        self._target_speed = simulator_state.velocity_mach #speed up a little
        self._tolerance_speed = None #0.1
        self._target_heading = torch.tensor(simulator_state.attitude_psi_rad, dtype=torch.float32, device=self.device)
        self._tolerance_heading = None #5 * torch.pi / 180 #three degrees
        self._target_roll = torch.tensor(0., dtype=torch.float32, device=self.device)
        self._tolerance_roll = None

        self._error_scale_alt = torch.tensor(350.0, dtype=torch.float32, device=self.device)  # m'
        self._error_scale_speed = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        self._error_scale_heading = torch.ones(1, dtype=torch.float32, device=self.device) * torch.pi/2
        self._error_scale_roll = torch.tensor(0.25, dtype=torch.float32, device=self.device)
        

        self._error_scale_smoothness_pdot = torch.tensor(0.1, dtype=torch.float32, device=self.device) 
        self._error_scale_smoothness_qdot = torch.tensor(0.1, dtype=torch.float32, device=self.device) 
        self._error_scale_smoothness_rdot = torch.tensor(0.1, dtype=torch.float32, device=self.device) 
        
        self._add_observations(simulator_state, td_out)
        self._add_last_action(primer_action, td_out)
        self._add_done_flags(simulator_state, td_out)

        return td_out
    
    def _set_seed(self, seed: Optional[int]):
        self.seed = seed