import os
import random
import time

import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torchrl.envs.utils import step_mdp
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.data.tensor_specs import TensorSpec, Composite
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    VecNorm,
    TransformedEnv,
    Compose,
    TensorDictPrimer,
    RenameTransform,
    InitTracker,
    StepCounter,
    RewardSum,
    RewardScaling
)
from torchrl.envs import (
    ParallelEnv,
    EnvCreator
)

from tensordict.nn import AddStateIndependentNormalScale
from tensordict.nn import TensorDictModule, TensorDictSequential, CudaGraphModule
from torchrl.envs import ExplorationType
from torchrl.modules.distributions import TanhNormal
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDict

from jsbsim_interface import AircraftSimulatorConfig, AircraftJSBSimSimulator, AircraftJSBSimInitialConditions
from control_env import JSBSimControlEnv, JSBSimControlEnvConfig
from transforms.euler_to_rotation_transform import EulerToRotation
from transforms.altitude_to_scale_code_transform import AltitudeToScaleCode
from transforms.altitude_to_digits_transform import AltitudeToDigits
from transforms.min_max_transform import TimeMinPool, TimeMaxPool
from transforms.episode_sum_transform import EpisodeSum
from transforms.difference_transform import Difference
from transforms.angular_difference_transform import AngularDifference
from transforms.planar_angle_cos_sin_transform import PlanarAngleCosSin

from hgauss.support_operator import SupportOperator
from hgauss.objectives.cliphgaussppo_loss import ClipHGaussPPOLoss

def log_trajectory(states, aircraft_uid, filename):
    with open(filename + ".acmi", mode='w', encoding='utf-8-sig') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
        timestamp = 0.0
        
        for state in states:
            f.write(f"#{timestamp:.2f}\n")
            timestamp += 1.0 / 60.0
            lat = state["lat"].item() * 180 / np.pi
            lon = state["lon"].item() * 180 / np.pi
            alt = state["alt"].item()
            roll, pitch, yaw = state["phi"].item(), state["theta"].item(), state["psi"].item()
            roll = roll * 180 / np.pi
            pitch = pitch * 180 / np.pi
            yaw = yaw * 180 / np.pi
            f.write(f"{aircraft_uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},")
            f.write(f"Name=JAS 39,")
            f.write(f"Color=Red")
            f.write(f"\n")
            

def make_models(cfg, observation_spec: TensorSpec, action_spec: TensorSpec, device: torch.device):
    input_shape = observation_spec.shape
    num_outputs = action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
        'safe_tanh': True
    }
    layer_width = 256

    policy_mlp_1 = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Tanh,
        out_features=layer_width,  # predict only loc
        num_cells=[layer_width],
        activate_last_layer=True,
        #norm_class=torch.nn.LayerNorm,
        #norm_kwargs=[{"elementwise_affine": False,
       #              "normalized_shape": hidden_size} for hidden_size in [512]],
    )

    # Initialize policy weights
    for layer in policy_mlp_1.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()


    # policy_mlp_2 = MLP(
    #     in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
    #     activation_class=torch.nn.Tanh,
    #     out_features=num_outputs,  # predict only loc
    #     num_cells=[layer_width,layer_width, layer_width],
    #     norm_class=torch.nn.LayerNorm,
    #     norm_kwargs=[{"elementwise_affine": False,
    #                  "normalized_shape": hidden_size} for hidden_size in [layer_width,layer_width, layer_width]],
    # )
        
    policy_mlp_2 = MLP(
        in_features=layer_width, #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[layer_width,layer_width],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                     "normalized_shape": hidden_size} for hidden_size in [layer_width,layer_width]],
    )
    # Initialize policy weights
    for layer in policy_mlp_2.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp_1,
        policy_mlp_2,
        AddStateIndependentNormalScale(
            action_spec.shape[-1], scale_lb=1e-8
        ),
    )
    
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation_vector"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=Composite(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    if cfg.ppo.loss_critic_type == "l2":
        value_mlp_1 = MLP(
            in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
            activation_class=torch.nn.Tanh,
            out_features=256,  # predict only loc
            num_cells=[256],
            activate_last_layer=True
        )

        for layer in value_mlp_1.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)
                layer.bias.data.zero_()

        value_mlp_2 = MLP(
            in_features=256, #+ num_fourier_features * 5 - 5,
            activation_class=torch.nn.Tanh,
            out_features=1,  # predict only loc
            num_cells=[256, 256],
            norm_class=torch.nn.LayerNorm,
            norm_kwargs=[{"elementwise_affine": False,
                        "normalized_shape": hidden_size} for hidden_size in [256, 256]],
        )

        for layer in value_mlp_2.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)
                layer.bias.data.zero_()

        value_mlp = torch.nn.Sequential(
            value_mlp_1,
            value_mlp_2,
        )

        value_module = ValueOperator(
            value_mlp,
            in_keys=["observation_vector"],
        )
        policy_module = policy_module.to(device)
        value_module = value_module.to(device)
        return policy_module, value_module

    elif cfg.ppo.loss_critic_type == "hgauss":
        layer_width = 2048
        nbins = cfg.network.nbins
        Vmin = cfg.network.vmin
        Vmax = cfg.network.vmax
        support = torch.linspace(Vmin, Vmax, nbins)

        value_mlp_1 = MLP(
            in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
            activation_class=torch.nn.Tanh,
            out_features=layer_width,  # predict only loc
            num_cells=[layer_width],
            activate_last_layer=True
        )

        for layer in value_mlp_1.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()

        value_mlp_2 = MLP(
            in_features=layer_width, #+ num_fourier_features * 5 - 5,
            activation_class=torch.nn.Tanh,
            out_features=nbins,  # predict only loc
            num_cells=[layer_width, layer_width],
            norm_class=torch.nn.LayerNorm,
            norm_kwargs=[{"elementwise_affine": False,
                        "normalized_shape": hidden_size} for hidden_size in [layer_width, layer_width]],
        )

        for layer in value_mlp_2.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()

        value_mlp = torch.nn.Sequential(
            value_mlp_1,
            value_mlp_2,
        )

        in_keys = ["observation_vector"]
        value_module_1 = TensorDictModule(
            in_keys=in_keys,
            out_keys=["state_value_logits"],
            module=value_mlp,
        )
        support_network = SupportOperator(support)
        value_module_2 = TensorDictModule(support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
        value_module = TensorDictSequential(value_module_1, value_module_2)
        support = support.to(device)
        policy_module = policy_module.to(device)
        value_module = value_module.to(device)
        return policy_module, value_module, support



def make_raw_environment():
    env = JSBSimControlEnv()
    return env

def apply_env_transforms(env):
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            #StepCounter(max_steps=2000),
            TimeMinPool(in_keys="mach", out_keys="episode_min_mach", T=2000),
            TimeMaxPool(in_keys="mach", out_keys="episode_max_mach", T=2000),
            RewardScaling(loc=0.0, scale=0.01, in_keys=["v_north", "v_east", "v_down", "udot", "vdot", "wdot"]),
            EulerToRotation(in_keys=["psi", "theta", "phi"], out_keys=["rotation"]),
            AltitudeToScaleCode(in_keys=["alt", "target_alt"], out_keys=["alt_code", "target_alt_code"], add_cosine=False),
            Difference(in_keys=["target_alt_code", "alt_code", "target_speed", "mach"], out_keys=["altitude_error", "speed_error"]),
            PlanarAngleCosSin(in_keys=["psi"], out_keys=["psi_cos_sin"]),
            AngularDifference(in_keys=["target_heading", "psi"], out_keys=["heading_error"]),                        
            CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "psi_cos_sin", "rotation", "v_north", "v_east", "v_down", "udot", "vdot", "wdot",
                                "p", "q", "r", "pdot", "qdot", "rdot", "last_action"],
                                    out_key="observation_vector", del_keys=False),        
            CatFrames(N=2, dim=-1, in_keys=["observation_vector"]),
            RewardSum(in_keys=["reward", "task_reward", "smoothness_reward"]),
            EpisodeSum(in_keys=["pdot", "p", "qdot", "q", "rdot", "r"])
        )
    )
    return env

def make_environment():
    env = make_raw_environment()
    env = apply_env_transforms(env)
    return env

def env_maker(cfg):
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: make_raw_environment()),
    )
    parallel_env = apply_env_transforms(parallel_env)
    return parallel_env

def load_model_state(model_name, run_folder_name=""):
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    load_from_saved_models = run_folder_name == ""
    if load_from_saved_models:
        outputs_folder = "../../../saved_models/"
    else:
        outputs_folder = "../../"

    model_load_filename = f"{model_name}.pt"
    load_model_dir = outputs_folder + run_folder_name
    print('Loading model from ' + load_model_dir)
    loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
    return loaded_state

@hydra.main(version_base="1.1", config_path="configs", config_name="main")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    torch.manual_seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)
    random.seed(cfg.random.seed)

    os.mkdir("ac_logging")


    template_env = make_environment()
    if cfg.ppo.loss_critic_type == "l2":
        policy_module, value_module, policy_parameters = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
    elif cfg.ppo.loss_critic_type == "hgauss":
        policy_module, value_module, support = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)

    load_model = True
    if load_model:
        model_dir="2024-11-30/15-31-45/"
        model_name = "training_snapshot_79040000"
        loaded_state = load_model_state(model_name, model_dir)

        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        policy_module.load_state_dict(actor_state)
        value_module.load_state_dict(critic_state)

    
    start_time = time.time()
    num_episodes = 5
    cfg.collector.env_per_collector = num_episodes

    envs = env_maker(cfg)

    print("Evaluating episodes")
    policy_module.eval()
    #3 minute flight = 3 * 60 = 180 seconds = 180 * 60 = 10800 time steps 
    episodes = envs.rollout(max_steps = 10800, policy=policy_module, break_when_any_done=False)
    #assert torch.sum(episodes['next', 'done']).item() == num_episodes, "Too many episodes"
    done_indices = episodes["next", "done"][..., 0].unsqueeze(-1)
    episode_rewards = episodes["next", "episode_reward"][done_indices]
    episode_smoothness_rewards = episodes["next", "episode_smoothness_reward"][done_indices]
    episode_task_rewards = episodes["next", "episode_task_reward"][done_indices]
   
    print("Average episode reward: ", episode_rewards.mean())
    print("Average episode task reward: ", episode_task_rewards.mean())
    print("Average episode smoothness reward: ", episode_smoothness_rewards.mean())

    for i, episode in enumerate(episodes.unbind(0)):
        log_trajectory(episode, "A0100", "trajectory_log_" + str(i))
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__=="__main__":
    main()
    