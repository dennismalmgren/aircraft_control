from typing import Union, Optional, List, Dict
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
from torchrl.envs import MarlGroupMapType
from tensordict.utils import NestedKey


from jsbsim_interface import AircraftJSBSimSimulator, AircraftSimulatorConfig, AircraftJSBSimInitialConditions, SimulatorState
from curriculum.curriculum_manager_jsbsim import CurriculumManagerJsbSim


DeviceType = Union[torch.device, str, int]

@dataclass
class JSBSimControlEnvConfig:
    agent_uids: List[str]
    agent_teams: List[str]

class JSBSimCombatEnv(EnvBase):
    def __init__(self,
                 *,
                 device: DeviceType = None,
                 batch_size: Optional[torch.Size] = None,
                 run_type_checks: bool = False,
                 allow_done_after_reset: bool = False):
        super().__init__(device=device, batch_size=batch_size, run_type_checks=run_type_checks, allow_done_after_reset=allow_done_after_reset)
        self.agents = {
            0: "player_0",
            1: "player_1"
        }
        self.agent_names: List[str] = ["player_0","player_1"]
        self.agent_names_to_indices_map = {
            "player_0": 0,
            "player_1": 1
        }
        self.group_map = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(self.agent_names)
        action_spec = Composite(device=self.device)
        observation_spec = Composite(device=self.device)
        reward_spec = Composite(device=self.device)
        done_spec = Composite(device=self.device)

        config = AircraftSimulatorConfig(jsbsim_module_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../py_modules/JSBSim'))
        self.aircraft_simulators = [AircraftJSBSimSimulator(config) for name in self.agent_names]
        for group in self.group_map.keys():

            # Placeholders
            aircraft_missile_count = 4
            observation_spec_single = Composite(
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

                #ownship
                available_missiles = Unbounded(shape=(1,), device=device, dtype=torch.int32),
                own_missile_locations = Unbounded(shape=(aircraft_missile_count, 3,), device=device, dtype=torch.float32),
                own_missile_velocities = Unbounded(shape=(aircraft_missile_count, 3, ), device=device, dtype=torch.float32),
                active_own_missile_masks = Unbounded(shape=(aircraft_missile_count, 1,), device=device, dtype=torch.bool),

                #sensors
                opponent_location = Unbounded(shape=(3,), device=device, dtype=torch.float32),
                opponent_velocity = Unbounded(shape=(3,), device=device, dtype=torch.float32),
                opponent_information_quality = Unbounded(shape=(1,), device=device, dtype=torch.float32),
                missile_location = Unbounded(shape=(aircraft_missile_count, 3,), device=device, dtype=torch.float32),
                missile_velocity = Unbounded(shape=(aircraft_missile_count, 3,), device=device, dtype=torch.float32),
                active_missile_masks = Unbounded(shape=(aircraft_missile_count, 1,), device=device, dtype=torch.bool),
            )
            group_observation_spec = torch.stack([observation_spec_single], dim=0)

            #aileron, elevator, rudder, throttle
            action_spec_single = Composite(
                                    control=Bounded(low = torch.tensor([-1.0, -1.0, -1.0, 0.0]),
                                    high = torch.tensor([1.0, 1.0, 1.0, 2.0]),
                                    device=device,
                                    dtype=torch.float32),
                                    fire=Categorical(n=2,
                                                    device=device))
            group_action_spec = torch.stack([action_spec_single], dim=0)

            group_done_spec = Composite(
                            terminated=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                            truncated=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                            done=Categorical(shape=(1,), n=2, device=device, dtype=torch.bool),
                        )
            
            reward_spec_single = Composite(
                reward = Unbounded(shape=(1,), device=device, dtype=torch.float32),
            )
            group_reward_spec = torch.stack([reward_spec_single], dim=0)

            action_spec[group] = group_action_spec
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec
            done_spec[group] = group_done_spec

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.state_spec = self.observation_spec.clone()
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

    @property
    def reward_keys(self) -> List[NestedKey]:
        return [("player_0", "reward"), ("player_1", "reward")]
    
    @property
    def action_keys(self) -> List[NestedKey]:
        return [("player_0", "action"), ("player_1", "action")]
    
    @property
    def done_keys(self) -> List[NestedKey]:
        return [("player_0", "done"), ("player_1", "done"),
                ("player_0", "terminated"), ("player_1", "terminated"),
                ("player_0", "truncated"), ("player_1", "truncated"),]
    
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
            task_reward = math.pow(alt_reward * speed_reward * heading_reward, 1/3)
            if math.isclose(task_reward, 1.0):
                smoothness_p_scale = 0.25
                smoothness_p_reward = math.exp(-((simulator_state.velocity_p_rad_sec / smoothness_p_scale)**2))
                smoothness_q_scale = 0.25
                smoothness_q_reward = math.exp(-((simulator_state.velocity_q_rad_sec / smoothness_q_scale)**2))
                smoothness_r_scale = 0.25
                smoothness_r_reward = math.exp(-((simulator_state.velocity_r_rad_sec / smoothness_r_scale)**2))
                smoothness_reward = math.pow(smoothness_p_reward * smoothness_q_reward * smoothness_r_reward, 1/3)
            total_reward = task_reward + smoothness_reward

        td_out.set("smoothness_reward", torch.tensor(smoothness_reward, device=self.device))
        td_out.set("task_reward", torch.tensor(task_reward, device=self.device))
        td_out.set("reward", torch.tensor(total_reward, device=self.device))
        
    def _evaluate_terminated(self, simulator_state: SimulatorState) -> bool:
        if simulator_state.position_h_sl_m < 300:
            return True
        else:
            return False
    
    def _add_done_flags(self, simulator_states: List[SimulatorState], tensordict: TensorDict):
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
        # if "aircraft_ic" in kwargs:
        #     aircraft_ic = kwargs["aircraft_ic"]
        # else:
        aircraft_ic_list = [self.curriculum_manager.get_initial_conditions() for _ in self.agent_names]

        primer_action = torch.zeros((self.action_spec.shape[-1],), device=self.device)
        aircraft_simulator_states = []
        for aircraft_ic, aircraft_simulator in zip(aircraft_ic_list, self.aircraft_simulators):
            aircraft_simulator_state = aircraft_simulator.reset(aircraft_ic)
            aircraft_simulator_states.append(aircraft_simulator_state)

        #self._target_altitude = simulator_state.position_h_sl_m #will be ignored.
        #self._tolerance_altitude = 40
        #self._target_speed = simulator_state.velocity_mach #speed up a little
        #self._tolerance_speed = 0.05
        #self._target_heading = simulator_state.attitude_psi_rad
        #self._tolerance_heading = 3 * torch.pi / 180 #three degrees
        self._add_observations(aircraft_simulator_states, td_out)
        self._add_last_action(primer_action, td_out)
        self._add_done_flags(aircraft_simulator_states, td_out)

        return td_out
    
    def _set_seed(self, seed: Optional[int]):
        self.seed = seed