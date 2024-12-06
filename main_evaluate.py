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


class SoftmaxLayer(torch.nn.Module):
    def __init__(self,  internal_dim: int):
        super().__init__()
        self.internal_dim = internal_dim

    def forward(self, x):
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, self.internal_dim)
        x = torch.softmax(x, dim=-1)
        return x.view(x_shape)    
    
class ClampOperator(torch.nn.Module):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, x):
        return torch.clamp(x, self.vmin, self.vmax)  

def make_models(cfg, observation_spec: TensorSpec, action_spec: TensorSpec, device: torch.device):
    input_shape = observation_spec.shape
    num_outputs = action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
       # 'safe_tanh': False
    }

    enc_dim = 4096
    latent_dim = 512
    softmax_dim = 8
    dynamics_dim = 512
    policy_dim = 512
    value_dim = 512

    softmax_activation_kwargs = {
        "internal_dim":softmax_dim
    }
    encoder_net = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=SoftmaxLayer,
        activation_kwargs=softmax_activation_kwargs,
        out_features=latent_dim,  # predict only loc
        num_cells=[enc_dim, latent_dim],
        activate_last_layer=True,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [enc_dim, latent_dim, latent_dim]],
    )

    dynamics_net = MLP(
        in_features=latent_dim + num_outputs,
        activation_class=SoftmaxLayer,
        activation_kwargs=softmax_activation_kwargs,
        out_features=latent_dim,
        num_cells=[dynamics_dim, dynamics_dim],
        activate_last_layer=True,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [dynamics_dim, dynamics_dim, latent_dim]],
    )

    policy_net = MLP(
        in_features=latent_dim,
        activation_class=torch.nn.Mish,
        out_features=num_outputs,
        num_cells=[policy_dim, policy_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [policy_dim, policy_dim]],
    )

    # Initialize weights
    for layer in encoder_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    for layer in dynamics_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    for layer in policy_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
    
    policy_net = torch.nn.Sequential(
        policy_net,
        AddStateIndependentNormalScale(
            action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    encoder_module = TensorDictModule(
        module=encoder_net,
        in_keys=["observation_vector"],
        out_keys=["observation_encoded"]
    )

    dynamics_module = TensorDictModule(
        module=dynamics_net,
        in_keys=["observation_encoded", "action"],
        out_keys=["next_observation_predicted"]
    )

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["observation_encoded"],
        out_keys=["loc", "scale"]
    )

    actor_module = ProbabilisticActor(
        module=TensorDictSequential(
            encoder_module,
            policy_module
        ),
        in_keys=["loc", "scale"],
        spec=Composite(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    learning_actor_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        spec=Composite(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    nbins = cfg.network.nbins
    Vmin = cfg.network.vmin
    Vmax = cfg.network.vmax
    support = torch.linspace(Vmin, Vmax, nbins)

    value_net = MLP(
        in_features=latent_dim, 
        activation_class=torch.nn.Mish,
        out_features=nbins,  
        num_cells=[value_dim, value_dim],
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [value_dim, value_dim]],
    )

    for layer in value_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    in_keys = ["observation_encoded"]
    value_module_1 = TensorDictModule(
        in_keys=in_keys,
        out_keys=["state_value_logits"],
        module=value_net,
    )
    support_network = SupportOperator(support)
    value_module_2 = TensorDictModule(support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
    value_module = TensorDictSequential(value_module_1, value_module_2)
    actor_module = actor_module.to(device)
    value_module = value_module.to(device)
    support = support.to(device)
    encoder_module = encoder_module.to(device)
    dynamics_module = dynamics_module.to(device)
    policy_module = policy_module.to(device)
    learning_actor_module = learning_actor_module.to(device)
    return actor_module, value_module, support, encoder_module, dynamics_module, policy_module, learning_actor_module



def make_raw_environment():
    env = JSBSimControlEnv()
    return env

def apply_env_transforms(env):
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=4000),
            TimeMinPool(in_keys="mach", out_keys="episode_min_mach", T=2000),
            TimeMaxPool(in_keys="mach", out_keys="episode_max_mach", T=2000),
            RewardScaling(loc=0.0, scale=0.01, in_keys=["u", "v", "w", "udot", "vdot", "wdot"]),
            #Lets try 3d-encoding later
            #AltitudeToScaleCode(in_keys=["u", "v", "w", "udot", "vdot", "wdot"], 
            #                    out_keys=["u_code", "v_code", "w_code", "udot_code", "vdot_code", "wdot_code"], 
            #                                add_cosine=True, num_wavelengths=11),
            EulerToRotation(in_keys=["psi", "theta", "phi"], out_keys=["rotation"]),
            AltitudeToScaleCode(in_keys=["alt", "target_alt"], out_keys=["alt_code", "target_alt_code"], add_cosine=False),
            Difference(in_keys=["target_alt_code", "alt_code", "target_speed", "mach"], out_keys=["altitude_error", "speed_error"]),
            PlanarAngleCosSin(in_keys=["psi"], out_keys=["psi_cos_sin"]),
            AngularDifference(in_keys=["target_heading", "psi"], out_keys=["heading_error"]),                        

            #CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "psi_cos_sin", "rotation", "u", "v", "w", "udot", "vdot", "wdot",
            #                    "p", "q", "r", "pdot", "qdot", "rdot", "last_action"],
            #                        out_key="observation_vector", del_keys=False),    
            CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "psi_cos_sin", "rotation", 
                                "u", "v", "w", "udot", "vdot", "wdot",
                    "p", "q", "r", "pdot", "qdot", "rdot", "last_action"],
            out_key="observation_vector", del_keys=False),        
            #CatFrames(N=60, dim=-1, in_keys=["observation_vector"]),
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
    actor_module, value_module, support, encoder_module, dynamics_module, policy_module, learning_actor_module = \
        make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)

    load_model = True
    if load_model:
        model_dir="2024-12-04/21-18-40/"
        model_name = "training_snapshot_79040000"
        loaded_state = load_model_state(model_name, model_dir)
        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        dynamics_state = loaded_state['model_dynamics']
        encoder_state = loaded_state["model_encoder"]
        policy_module.load_state_dict(actor_state)
        value_module.load_state_dict(critic_state)
        dynamics_module.load_state_dict(dynamics_state)
        encoder_module.load_state_dict(encoder_state)
    
    start_time = time.time()
    num_episodes = 5
    cfg.collector.env_per_collector = num_episodes

    envs = env_maker(cfg)

    print("Evaluating episodes")
    policy_module.eval()
    #3 minute flight = 3 * 60 = 180 seconds = 180 * 60 = 10800 time steps 
    episodes = envs.rollout(max_steps = 4000, policy=actor_module, break_when_any_done=False)
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
    