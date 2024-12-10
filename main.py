import os
import random
import time
import math

import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
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
from tensordict.nn.probabilistic import InteractionType, set_interaction_type
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
from transforms.sum_transform import Sum
from transforms.angular_difference_transform import AngularDifference
from transforms.planar_angle_cos_sin_transform import PlanarAngleCosSin

from hgauss.support_operator import SupportOperator
from hgauss.objectives.cliphgauss_worldmodel_ppo_loss import ClipHGaussWorldModelPPOLoss

def log_trajectory(states, aircraft_uid):
    with open("thelog.acmi", mode='w', encoding='utf-8-sig') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
        timestamp = 0.0
        
        for state in states:
            f.write(f"#{timestamp:.2f}\n")
            timestamp += 1.0 / 60.0
            lat = state["lat"] * 180 / np.pi
            lon = state["lon"] * 180 / np.pi
            alt = state["alt"]
            roll, pitch, yaw = state["phi"], state["theta"], state["psi"]
            roll = roll * 180 / np.pi
            pitch = pitch * 180 / np.pi
            yaw = yaw * 180 / np.pi
            f.write(f"{aircraft_uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},")
            f.write(f"Name=JAS 39,")
            f.write(f"Color=Red")
            f.write(f"\n")

class PlanningModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, estimated_value: torch.Tensor, action: torch.Tensor):
        action_index = estimated_value.argmax(dim=-2, keepdim=True)
        action_index = action_index.expand(-1, -1, 4)
        action_sampled = action.gather(-2, action_index).squeeze(-2)
        return action_sampled
    
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

    reward_net_1 = MLP(
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

    reward_net_2 = MLP(
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

    for layer in dynamics_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    for layer in policy_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
    
    for layer in reward_net_1.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
    
    for layer in reward_net_2.modules():
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

   
    value_names = ["safety", "scale_1", "scale_2"]
    vmins = [cfg.network.vmin_1, cfg.network.vmin_2, cfg.network.vmin_3]
    vmaxs = [cfg.network.vmax_1, cfg.network.vmax_2, cfg.network.vmax_3]
    rmins = [cfg.network.rmin_1, cfg.network.rmin_2, cfg.network.rmin_3]
    rmaxs = [cfg.network.rmax_1, cfg.network.rmax_2, cfg.network.rmax_3]
    nbins = cfg.network.nbins

    value_modules = []
    value_supports = []

    for value_name, Vmin, Vmax in zip(value_names, vmins, vmaxs):
        dk = (Vmax - Vmin) / (nbins - 4)
        Ktot = dk * nbins
        Vmax = math.ceil(Vmin + Ktot)

        value_support = torch.linspace(Vmin, Vmax, nbins)
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

        value_module_1 = TensorDictModule(
            in_keys=["observation_encoded"],
            out_keys=[f"state_value_{value_name}_logits"],
            module=value_net,
        )
        support_network = SupportOperator(value_support)
        value_module_2 = TensorDictModule(support_network, in_keys=[f"state_value_{value_name}_logits"], out_keys=[f"state_value_{value_name}"])
        value_module = TensorDictSequential(value_module_1, value_module_2)
        value_module = value_module.to(device)
        value_modules.append(value_module)
        value_support = value_support.to(device)
        value_supports.append(support)

        reward_modules = []
        reward_module_1 = TensorDictModule(
            module=reward_net_1,
            in_keys=["observation_encoded", "action"],
            out_keys=["next_reward_scale_1_predicted"]
        )
        reward_module_2 = TensorDictModule(
            module=reward_net_2,
            in_keys=["observation_encoded", "action"],
            out_keys=["next_reward_scale_2_predicted"]
        )
        reward_modules.append(reward_module_1)
        reward_modules.append(reward_module_2)
        reward_module = TensorDictSequential(*reward_modules)


    value_module_final = TensorDictModule(
        module=lambda x, y: x + y,
        in_keys=[f"state_value_{value_name}" for value_name in value_names],
        out_keys=["state_value"]
    )
    
    value_module = TensorDictSequential(
        [*value_modules, value_module_final]
    )

    value_module = value_module.to(device)
    actor_module = actor_module.to(device)
    support = support.to(device)
    encoder_module = encoder_module.to(device)
    dynamics_module = dynamics_module.to(device)
    reward_module = reward_module.to(device)
    policy_module = policy_module.to(device)
    latent_actor_module = latent_actor_module.to(device)

    return actor_module, value_modules, value_module, support, encoder_module, dynamics_module, policy_module, latent_actor_module, reward_module, reward_modules



def make_raw_environment():
    env = JSBSimControlEnv()
    return env

def apply_env_transforms(env, cfg, is_train = True):
    reward_keys = list(env.reward_spec.keys())
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=cfg.env.max_time_steps_train if is_train else cfg.env.max_time_steps_eval),
            AltitudeToScaleCode(in_keys=["u", "v", "w", "udot", "vdot", "wdot"], 
                                out_keys=["u_code", "v_code", "w_code", "udot_code", "vdot_code", "wdot_code"], 
                                            add_cosine=False, base_scale=0.1),
            EulerToRotation(in_keys=["psi", "theta", "phi"], out_keys=["rotation"]),
            AltitudeToScaleCode(in_keys=["alt", "target_alt"], out_keys=["alt_code", "target_alt_code"], add_cosine=False),
            Difference(in_keys=["target_alt_code", "alt_code", "target_speed", "mach"], out_keys=["altitude_error", "speed_error"]),
            PlanarAngleCosSin(in_keys=["psi"], out_keys=["psi_cos_sin"]),
            AngularDifference(in_keys=["target_heading", "psi"], out_keys=["heading_error"]),                        

            CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "rotation", 
                                "u_code", "v_code", "w_code", "udot_code", "vdot_code", "wdot_code",
                    "p", "q", "r", "pdot", "qdot", "rdot", "last_action"],
            out_key="observation_vector", del_keys=False),        
            RewardSum(in_keys=reward_keys),
        )
    )
    return env

def make_environment(cfg, is_train=True):
    env = make_raw_environment()
    env = apply_env_transforms(env, cfg, is_train)
    return env

def env_maker_eval(cfg):
    parallel_env = ParallelEnv(
        cfg.num_eval_envs,
        EnvCreator(lambda cfg=cfg: make_raw_environment()),
    )
    parallel_env = apply_env_transforms(parallel_env, cfg, is_train=False)
    return parallel_env

def env_maker(cfg):
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: make_raw_environment()),
    )
    parallel_env = apply_env_transforms(parallel_env, cfg)
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
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    use_torch_compile = cfg.use_torch_compile

    torch.manual_seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)
    random.seed(cfg.random.seed)

    os.mkdir("ac_logging")

    exp_name = generate_exp_name("AC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ac_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode,
                          "project": cfg.logger.project,},
        )

    template_env = make_environment(cfg)
    #test_td = template_env.reset()

    actor_module, value_modules, value_module, support, encoder_module, dynamics_module, policy_module, latent_actor_module, reward_module, reward_modules = \
        make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
    loss_module = ClipHGaussWorldModelPPOLoss(
        actor_network=learning_actor_module,
        critic_networks=value_modules,
        encoder_network=encoder_module,
        dynamics_network=dynamics_module,
        reward_network=reward_module,
        clip_epsilon=cfg.ppo.clip_epsilon,
        loss_critic_type=cfg.ppo.loss_critic_type,
        entropy_coef=cfg.ppo.entropy_coef,
        normalize_advantage= True,
        support=supports
    )
    adv_modules = []
    for value_module in 
    adv_module = GAE(
        gamma=cfg.ppo.gamma,
        lmbda=cfg.ppo.gae_lambda,
        value_network=value_module,
    )

    actor_optim = torch.optim.AdamW(policy_module.parameters(), lr=cfg.optim.lr_policy, eps=cfg.optim.eps)
    critic_optim = torch.optim.AdamW(value_module.parameters(), lr=cfg.optim.lr_value, eps=cfg.optim.eps)
    consistency_optim = torch.optim.AdamW(list(encoder_module.parameters()) + 
                                          list(dynamics_module.parameters()) +
                                          list(reward_module.parameters()), lr=cfg.optim.lr_policy, eps=cfg.optim.eps)

    collected_frames = 0
    cfg_logger_save_interval = cfg.logger.save_interval
    cfg_logger_eval_interval = cfg.logger.eval_interval
    loaded_frames = 0

    load_model = False
    if load_model:
        model_dir="2024-12-07/10-39-45/"
        model_name = "training_snapshot_8040000"
        loaded_state = load_model_state(model_name, model_dir)

        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        dynamics_state = loaded_state['model_dynamics']
        encoder_state = loaded_state['model_encoder']
        reward_state = loaded_state['model_reward']
        actor_optim_state = loaded_state['actor_optimizer']
        critic_optim_state = loaded_state['critic_optimizer']
        consistency_optim_state = loaded_state['consistency_optimizer']
        collected_frames = loaded_state['collected_frames']['collected_frames']
        loaded_frames = collected_frames
        policy_module.load_state_dict(actor_state)
        value_module.load_state_dict(critic_state)
        dynamics_module.load_state_dict(dynamics_state)
        encoder_module.load_state_dict(encoder_state)
        reward_module.load_state_dict(reward_state)
        actor_optim.load_state_dict(actor_optim_state)
        critic_optim.load_state_dict(critic_optim_state)
        consistency_optim.load_state_dict(consistency_optim_state)


    frames_remaining = cfg.collector.total_frames - collected_frames

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=env_maker(cfg),
        policy=actor_module,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=frames_remaining,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
      #  compile_policy=True
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.optim.mini_batch_size,
    )
    
    #create eval envs
    eval_envs = env_maker_eval(cfg)

    reward_keys = list(eval_envs.reward_spec.keys())
    cfg_loss_ppo_epochs = cfg.ppo.epochs
    cfg_max_grad_norm = cfg.optim.max_grad_norm
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, ncols=0)
    pbar.update(collected_frames)
    sampling_start = time.time()
    num_mini_batches = cfg.collector.frames_per_batch // cfg.optim.mini_batch_size
    total_network_updates = (
        (frames_remaining // cfg.collector.frames_per_batch)
        * cfg_loss_ppo_epochs
        * num_mini_batches
    )

    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])
    norms = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])
    # K = 100
    # def planner(actor_module, encoder_module, dynamics_module, reward_module, value_module):
    #     def plan(observation_vector: torch.Tensor):
    #         observation_vector = observation_vector.expand(K, -1, -1)
    #         observation_encoded, loc, var, action, log_prob = actor_module(observation_vector)
    #         next_states = dynamics_module(observation_encoded, action)
    #         next_reward = reward_module(observation_encoded, action) #todo: incorporate next state into the reward prediction?
    #         next_value_logits, next_value_value = value_module(next_states)
    #         value_estimated = next_reward + 0.995 * next_value_value
    #         action_index = value_estimated.argmax(dim=0)
    #         action_index = action_index.unsqueeze(0).expand(-1, -1, 4)
    #         action_sampled = action.gather(0, action_index).squeeze(0)
    #         return action_sampled
        
    #     planning_module = TensorDictModule(
    #         module=plan,
    #         in_keys=["observation_vector"],
    #         out_keys=["action"]
    #     )
    #     return planning_module
    
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        #episode_pdot = data["next", "episode_pdot"][data["next", "done"]]
        #episode_p = data["next", "episode_p"][data["next", "done"]]
        #episode_min_mach = data["next", "episode_min_mach"][data["next", "done"]]
        #episode_max_mach = data["next", "episode_max_mach"][data["next", "done"]]
        if len(episode_rewards) > 0:
            #episode_length = data["next", "step_count"][data["next", "done"]]
            # log_info.update(
            #     {
            #         "train/pdot": (episode_pdot / episode_length).mean().item(),
            #         "train/p": (episode_p / episode_length).mean().item(),
            #         "train/episode_length": episode_length.float().mean().item(), #mean?
            #      #   "train/episode_min_mach": episode_min_mach.mean().item(),
            #     #    "train/episode_max_mach": episode_max_mach.mean().item(),
            #     }
            # )
            for reward_key in reward_keys:
                log_info.update(
                    {
                        f"train/{reward_key}": data["next", "episode_" + reward_key][
                            data["next", "done"]
                        ].mean().item()
                    }
                )
        data = data.select("observation_vector", 
                           "observation_encoded",
                           "action", 
                           "sample_log_prob",
                           ("next", "reward"), 
                           ("next", "terminated"), 
                           ("next","done"), 
                           ("next", "observation_vector"),)
        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            # Compute GAE

            with torch.no_grad():
                next_step = data["next"]
                encoder_module(data)
                encoder_module(next_step)
                data["next"] = next_step
                data = adv_module(data)
            data_extend = data.reshape(-1)
            data_buffer.extend(data_extend)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective", "loss_consistency", "loss_reward"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]
                consistency_loss = loss["loss_consistency"] * 20
                reward_loss = loss["loss_reward"]
                consistency_critic_loss = consistency_loss + critic_loss + reward_loss

                consistency_optim.zero_grad()
                critic_optim.zero_grad()                
                consistency_critic_loss.backward()
                consistency_grad_norm = torch.nn.utils.clip_grad_norm_(list(encoder_module.parameters())
                                                                        + list(dynamics_module.parameters()) + 
                                                                        list(reward_module.parameters()), cfg_max_grad_norm * 2)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(value_module.parameters(), cfg_max_grad_norm)
                critic_optim.step()
                consistency_optim.step()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy_module.parameters(), cfg_max_grad_norm)
                actor_optim.step()

                #critic_loss.backward()
                norms[j, k] = TensorDict({
                    "actor_grad_norm": actor_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
                    "consistency_grad_norm": consistency_grad_norm
                })

         # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        norms_mean = norms.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in norms_mean.items():
            log_info.update({f"train/{key}": value.item()})

        log_info.update(
            {
                "train/lr_policy": cfg.optim.lr_policy,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": cfg.ppo.clip_epsilon
            }
        )
        if ((i - 1) * frames_in_batch + loaded_frames) // cfg_logger_eval_interval < \
           (i * frames_in_batch + loaded_frames) // cfg_logger_eval_interval:
            actor_module.eval()
            actor_module = actor_module.to("cpu")
            with set_interaction_type(InteractionType.DETERMINISTIC):
                eval_results = eval_envs.rollout(cfg.env.max_time_steps_eval, policy=actor_module)
            actor_module = actor_module.to(device)

            actor_module.train()
            eval_done_indices = eval_results["next", "done"][..., 0].unsqueeze(-1)
            if torch.sum(eval_done_indices.float()) > 0.0:                
                for reward_key in reward_keys:
                    log_info.update(
                        {
                            f"eval/{reward_key}": eval_results["next", "episode_" + reward_key][eval_done_indices
                            ].mean().item()
                        }
                    )

        if ((i - 1) * frames_in_batch + loaded_frames) // cfg_logger_save_interval < \
           (i * frames_in_batch + loaded_frames) // cfg_logger_save_interval:
                savestate = {
                        'model_actor': policy_module.state_dict(),
                        'model_critic': value_module.state_dict(),
                        'model_dynamics': dynamics_module.state_dict(),
                        'model_encoder': encoder_module.state_dict(),
                        'model_reward': reward_module.state_dict(),
                        'actor_optimizer': actor_optim.state_dict(),
                        'critic_optimizer': critic_optim.state_dict(),
                        'consistency_optimizer': consistency_optim.state_dict(),

                        "collected_frames": {"collected_frames": collected_frames},
                }
                torch.save(savestate, f"training_snapshot_{collected_frames}.pt")


        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)  

        collector.update_policy_weights_()
        sampling_start = time.time()


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__=="__main__":
    main()
    