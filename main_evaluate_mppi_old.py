import os
import random
import time
import math

import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torchrl.envs.utils import step_mdp
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.data.tensor_specs import TensorSpec, Composite, Bounded, Unbounded
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
from tensordict.nn.probabilistic import InteractionType, set_interaction_type

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
from torchrl.modules.tensordict_module import WorldModelWrapper
from torchrl.modules.planners import MPPIPlanner
from torchrl.envs.model_based import ModelBasedEnvBase

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
from transforms.observation_scaling_transform import ObservationScaling

from hgauss.support_operator import SupportOperator
from hgauss.objectives.cliphgaussppo_loss import ClipHGaussPPOLoss
from torchrl.objectives.value import TDLambdaEstimator

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
        "low": action_spec.space.low,
        "high": action_spec.space.high,
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

    decoder_net = MLP(
        in_features=latent_dim, #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Mish,
        #activation_kwargs=softmax_activation_kwargs,
        out_features=input_shape[-1],  # predict only loc
        num_cells=[latent_dim, latent_dim],
        activate_last_layer=False,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [latent_dim, latent_dim]],    
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

    reward_net = MLP(
        in_features=latent_dim + num_outputs,
        activation_class=SoftmaxLayer,
        activation_kwargs=softmax_activation_kwargs,
        out_features=1,
        num_cells=[dynamics_dim, dynamics_dim],
        activate_last_layer=False,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [dynamics_dim, dynamics_dim]],
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

    for layer in decoder_net.modules():
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
    
    for layer in reward_net.modules():
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

    decoder_module = TensorDictModule(
        module=decoder_net,
        in_keys=["observation_encoded"],
        out_keys=["observation_vector_decoded"]
    )

    dynamics_module = TensorDictModule(
        module=dynamics_net,
        in_keys=["observation_encoded", "action"],
        out_keys=["observation_encoded"]
    )

    reward_module = TensorDictModule(
        module=reward_net,
        in_keys=["observation_encoded", "action"],
        out_keys=["reward"]
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
        default_interaction_type=InteractionType.RANDOM,
    )

    latent_actor_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        spec=Composite(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    )

    nbins = cfg.network.nbins
    Vmin = cfg.network.vmin
    Vmax = cfg.network.vmax
    dk = (Vmax - Vmin) / (nbins - 4)
    Ktot = dk * nbins
    Vmax = math.ceil(Vmin + Ktot)

    support = torch.linspace(Vmin, Vmax, nbins)

    value_net = MLP(
        in_features=latent_dim, 
        activation_class=torch.nn.Mish,
        out_features=nbins,  
        num_cells=[value_dim, value_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [value_dim, value_dim, value_dim]],
    )

    for layer in value_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    value_module_1 = TensorDictModule(
        in_keys=["observation_encoded"],
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
    decoder_module = decoder_module.to(device)
    dynamics_module = dynamics_module.to(device)
    reward_module = reward_module.to(device)
    policy_module = policy_module.to(device)
    latent_actor_module = latent_actor_module.to(device)

    return actor_module, value_module, support, encoder_module, dynamics_module, policy_module, latent_actor_module, reward_module, decoder_module


def make_raw_environment():
    env = JSBSimControlEnv()
    return env

def apply_env_transforms(env, cfg, is_train = False):
    reward_keys = list(env.reward_spec.keys())
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=cfg.env.max_time_steps_train if is_train else cfg.env.max_time_steps_eval),
            Difference(in_keys=["target_alt", "alt", "target_speed", "mach", "target_heading", "psi"], out_keys=["altitude_error", "speed_error", "heading_error"]),
            ObservationScaling(in_keys=["target_alt", "alt", "u", "v", "w", "udot", "vdot", "wdot",
                                   "altitude_error"], 
                          out_keys=["target_alt_scaled", "alt_scaled", "u_scaled", "v_scaled", "w_scaled", "udot_scaled", "vdot_scaled", "wdot_scaled",
                                    "altitude_error_scaled"],
                          loc = 0.0, scale=0.001),
            CatTensors(in_keys=["altitude_error_scaled", "speed_error", "alt_scaled", "mach", 
                                "psi", "theta", "phi",
                                "u_scaled", "v_scaled", "w_scaled", "udot_scaled", "vdot_scaled", "wdot_scaled",
                                "p", "q", "r", "pdot", "qdot", "rdot", "heading_error"], out_key="observation_vector", del_keys=False),

            RewardSum(in_keys=reward_keys),
        )
    )
    return env

def make_environment(cfg, is_train=False):
    env = make_raw_environment()
    env = apply_env_transforms(env, cfg, is_train)
    return env

def env_maker(cfg):
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: make_raw_environment()),
    )
    parallel_env = apply_env_transforms(parallel_env, cfg, is_train=False)
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


    template_env = make_environment(cfg)
    actor_module, value_module, support, encoder_module, dynamics_module, policy_module, learning_actor_module, reward_module, decoder_module = \
        make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
    
    load_model = True
    if load_model:
        model_dir="2024-12-18/00-54-01/"
        model_name = "training_snapshot_36040000"
        loaded_state = load_model_state(model_name, model_dir)
        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        dynamics_state = loaded_state['model_dynamics']
        encoder_state = loaded_state["model_encoder"]
        decoder_state = loaded_state["model_decoder"]
        reward_state = loaded_state["model_reward"]
        policy_module.load_state_dict(actor_state)
        value_module.load_state_dict(critic_state)
        dynamics_module.load_state_dict(dynamics_state)
        encoder_module.load_state_dict(encoder_state)
        decoder_module.load_state_dict(decoder_state)
        reward_module.load_state_dict(reward_state)

        class PredictiveWorldModelWrapper(TensorDictSequential):
            """World model wrapper.

            This module wraps together a transition model and a reward model.
            The transition model is used to predict an imaginary world state.
            The reward model is used to predict the reward of the imagined transition.

            Args:
                transition_model (TensorDictModule): a transition model that generates a new world states.
                reward_model (TensorDictModule): a reward model, that reads the world state and returns a reward.

            """

            def __init__(
                self, transition_model: TensorDictModule, reward_model: TensorDictModule
            ):
                super().__init__(reward_model, transition_model)

            def get_transition_model_operator(self) -> TensorDictModule:
                """Returns a transition operator that maps either an observation to a world state or a world state to the next world state."""
                return self.module[0]

            def get_reward_operator(self) -> TensorDictModule:
                """Returns a reward operator that maps a world state to a reward."""
                return self.module[1]

        #todo: move latent_dim to config
        class ModelBasedEnv(ModelBasedEnvBase):
            def __init__(self, world_model, device="cpu", batch_size=None):
                super().__init__(world_model, device=device, batch_size=batch_size)
                self.observation_spec = Composite(
                    observation_encoded=Unbounded((512,)),
                    shape=batch_size
                )
                self.state_spec = Composite(
                    observation_encoded=Unbounded((512,)),
                    shape=batch_size
                )
                self.action_spec = Unbounded((4,))
                self.reward_spec = Unbounded((1,))

            def set_reset_state(self, tensordict: TensorDict):
                self._reset_tensordict = tensordict

            def _reset(self, tensordict: TensorDict) -> TensorDict:
                tensordict = TensorDict({},
                    batch_size=self.batch_size,
                    device=self.device,
                )
                tensordict = tensordict.update(self.state_spec.rand()) #how do we deal with resets? do we even?
                tensordict = tensordict.update(self.observation_spec.rand())
                return tensordict
#                return self._reset_tensordict.clone()
            
    world_model = PredictiveWorldModelWrapper(dynamics_module, reward_module)

    start_time = time.time()
    num_envs = cfg.collector.env_per_collector = 30
    conf_env = make_raw_environment()
    ic = [{"aircraft_ic": conf_env.curriculum_manager.get_initial_conditions()} for _ in range(num_envs)]

    envs = env_maker(cfg)
    print("Evaluating episodes")
    policy_module.eval()
    #3 minute flight = 3 * 60 = 180 seconds = 180 * 60 = 10800 time steps 
    episode_tds = []
    with set_interaction_type(InteractionType.DETERMINISTIC) and torch.no_grad():
        next_step_td_ = envs.reset(list_of_kwargs=ic)
        done = torch.zeros((num_envs, 1), dtype=torch.bool)
        while not torch.all(done):
            next_step_td_ = actor_module(next_step_td_)
            next_step_td, next_step_td_ = envs.step_and_maybe_reset(next_step_td_)
            done = done | next_step_td["next", "done"]
            episode_tds.append(next_step_td)
        episodes = torch.stack(episode_tds, dim=-1)
    done_indices = episodes["next", "done"][..., 0].unsqueeze(-1)
    episode_rewards = episodes["next", "episode_reward"][done_indices]
    episode_smoothness_rewards = episodes["next", "episode_reward_smoothness"][done_indices]
    episode_task_rewards = episodes["next", "episode_reward_task"][done_indices]
   
    print("Evaluation results (deterministic)")
    print("Average episode reward: ", episode_rewards.mean())
    print("Average episode task reward: ", episode_task_rewards.mean())
    print("Average episode smoothness reward: ", episode_smoothness_rewards.mean())


    with set_interaction_type(InteractionType.RANDOM) and torch.no_grad():
        next_step_td_ = envs.reset(list_of_kwargs=ic)
        done = torch.zeros((num_envs, 1), dtype=torch.bool)
        while not torch.all(done):
            next_step_td_ = actor_module(next_step_td_)
            next_step_td, next_step_td_ = envs.step_and_maybe_reset(next_step_td_)
            done = done | next_step_td["next", "done"]
            episode_tds.append(next_step_td)
        episodes = torch.stack(episode_tds, dim=-1)
    done_indices = episodes["next", "done"][..., 0].unsqueeze(-1)
    episode_rewards = episodes["next", "episode_reward"][done_indices]
    episode_smoothness_rewards = episodes["next", "episode_reward_smoothness"][done_indices]
    episode_task_rewards = episodes["next", "episode_reward_task"][done_indices]
   
    print("Evaluation results (random)")
    print("Average episode reward: ", episode_rewards.mean())
    print("Average episode task reward: ", episode_task_rewards.mean())
    print("Average episode smoothness reward: ", episode_smoothness_rewards.mean())

#    for i, episode in enumerate(episodes.unbind(0)):
#        log_trajectory(episode, "A0100", "trajectory_log_raw_" + str(i))
    n_envs = envs.batch_size[0]
    with torch.no_grad():
        episode_td = []
        next_step_td_ = envs.reset(list_of_kwargs=ic)
        done = torch.zeros((n_envs, 1), dtype=torch.bool)
        while not torch.all(done):
            dist = actor_module.get_dist(next_step_td_)
            next_step_td_ = value_module(next_step_td_)
            n_sample_actions = 16
            actions = dist.sample(sample_shape=(n_sample_actions,))
            actions = torch.cat((actions, dist.deterministic_sample.unsqueeze(0)), dim=0)
            n_final_actions = actions.shape[0]
            p_actions = dist.log_prob(actions).unsqueeze(-1)
            actions = actions.permute(1, 0, 2)
            p_actions = p_actions.permute(1, 0, 2)
            plan_td = next_step_td_.select("observation_encoded").unsqueeze(-1).expand(n_envs, n_final_actions)
            plan_td["action"] = actions
            plan_td = reward_module(plan_td)
            plan_td = dynamics_module(plan_td)
            plan_td = value_module(plan_td)
            #ignore terminals now)
            q_vals = plan_td["reward"] + 0.99 * plan_td["state_value"]# - next_step_td_["state_value"].unsqueeze(-2)
            res = 0.01 * q_vals + p_actions
            action_indices = torch.argmax(res, dim=1)
            action_td = torch.gather(plan_td, dim=1, index=action_indices)
            actions = action_td["action"][:, 0]
            next_step_td_["action"] = actions
            next_step_td, next_step_td_ = envs.step_and_maybe_reset(next_step_td_)
            done = done | next_step_td["next", "done"]
            episode_td.append(next_step_td)
    episodes = torch.stack(episode_td, dim=-1)
    done_indices = episodes["next", "done"][..., 0].unsqueeze(-1)
    episode_rewards = episodes["next", "episode_reward"][done_indices]
    episode_smoothness_rewards = episodes["next", "episode_reward_smoothness"][done_indices]
    episode_task_rewards = episodes["next", "episode_reward_task"][done_indices]
    print("Evaluation results (planner)")
    print("Average episode reward: ", episode_rewards.mean())
    print("Average episode task reward: ", episode_task_rewards.mean())
    print("Average episode smoothness reward: ", episode_smoothness_rewards.mean())
    print('ok')
    # with set_interaction_type(InteractionType.DETERMINISTIC):
    #     episodes = envs.rollout(max_steps = 2000, policy=encoder_planner, break_when_any_done=False)
    # done_indices = episodes["next", "done"][..., 0].unsqueeze(-1)
    # episode_rewards = episodes["next", "episode_reward"][done_indices]
    # episode_smoothness_rewards = episodes["next", "episode_smoothness_reward"][done_indices]
    # episode_task_rewards = episodes["next", "episode_task_reward"][done_indices]

    # print("Evaluation results (planner)")
    # print("Average episode reward (with planner): ", episode_rewards.mean())
    # print("Average episode task reward (with planner): ", episode_task_rewards.mean())
    # print("Average episode smoothness reward (with planner): ", episode_smoothness_rewards.mean())

#    for i, episode in enumerate(episodes.unbind(0)):
#        log_trajectory(episode, "A0100", "trajectory_log_planned_" + str(i))

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__=="__main__":
    main()
    