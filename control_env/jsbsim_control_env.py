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
#velocities/h-dot-fps 
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
            reward = Unbounded(shape=(1,), device=device, dtype=torch.float32), #start with the one.
            task_reward = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            smoothness_reward = Unbounded(shape=(1,), device=device, dtype=torch.float32),
        )
        self.state_spec = self.observation_spec.clone()

        self._target_altitude = None
        self._target_speed = None
        self._target_heading = None
        self._tolerance_altitude = None
        self._tolerance_speed = None
        self._tolerance_heading = None
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
        sin_diff = math.sin(theta2) * math.cos(theta1) - math.cos(theta2) * math.sin(theta1)
        cos_diff = math.cos(theta2) * math.cos(theta1) + math.sin(theta2) * math.sin(theta1)
        error = math.atan2(sin_diff, cos_diff)
        return error

    def _add_reward(self, simulator_state: SimulatorState, td_out):
        alt_reward = 0.0
        speed_reward = 0.0
        heading_reward = 0.0
        roll_reward = 0.0
        total_reward = 0.0
        smoothness_reward = 0.0
        task_reward = 0.0
        if simulator_state.position_h_sl_m < 300:
            task_reward = -100.0
            total_reward = task_reward
        else:
            alt_error_scale = 350.0  # m'
            alt_error = simulator_state.position_h_sl_m - self._target_altitude
            if self._tolerance_altitude is not None:
                alt_error = max(0, abs(alt_error) - self._tolerance_altitude)
            alt_reward = math.exp(-((alt_error / alt_error_scale) ** 2))
            
            speed_error_scale = 0.5
            speed_error = simulator_state.velocity_mach - self._target_speed
            if self._tolerance_speed is not None:
                speed_error = max(0, abs(speed_error) - self._tolerance_speed)

            speed_reward = math.exp(-((speed_error / speed_error_scale)**2))
            
            heading_error_scale = math.pi/2
            heading_error = self.heading_error(simulator_state.attitude_psi_rad, self._target_heading)
            if self._tolerance_heading is not None:
                heading_error = max(0, abs(heading_error) - self._tolerance_heading)

            heading_reward = math.exp(-((heading_error / heading_error_scale)**2))
            
            roll_error_scale = 0.25
            roll_error = simulator_state.attitude_phi_rad
            tolerance_roll = 5 * math.pi / 180.0
            roll_error = max(0, abs(roll_error) - tolerance_roll)
            roll_reward = math.exp(-((roll_error / roll_error_scale)**2))

            task_reward = math.pow(alt_reward * speed_reward * heading_reward * roll_reward, 1/4)


            if math.isclose(task_reward, 1.0):
                smoothness_p_scale = 0.1
                smoothness_p_reward = math.exp(-((simulator_state.acceleration_pdot_rad_sec2 / smoothness_p_scale)**2))
                smoothness_q_scale = 0.1
                smoothness_q_reward = math.exp(-((simulator_state.acceleration_qdot_rad_sec2 / smoothness_q_scale)**2))
                smoothness_r_scale = 0.1
                smoothness_r_reward = math.exp(-((simulator_state.acceleration_rdot_rad_sec2 / smoothness_r_scale)**2))
                
                smoothness_reward = math.pow(smoothness_p_reward * smoothness_q_reward * smoothness_r_reward , 1/3)

            total_reward = task_reward + smoothness_reward

        td_out.set("smoothness_reward", torch.tensor(smoothness_reward, device=self.device))
        td_out.set("task_reward", torch.tensor(task_reward, device=self.device))
        td_out.set("reward", torch.tensor(total_reward, device=self.device))
        
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
        self._target_altitude = simulator_state.position_h_sl_m #will be ignored.
        self._tolerance_altitude = 100
        self._target_speed = simulator_state.velocity_mach #speed up a little
        self._tolerance_speed = 0.1
        self._target_heading = simulator_state.attitude_psi_rad
        self._tolerance_heading = 5 * torch.pi / 180 #three degrees
        self._add_observations(simulator_state, td_out)
        self._add_last_action(primer_action, td_out)
        self._add_done_flags(simulator_state, td_out)

        return td_out
    
    def _set_seed(self, seed: Optional[int]):
        self.seed = seed