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
    RewardScaling,
    ClipTransform
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
from hgauss.objectives.cliphgaussppo_loss import ClipHGaussPPOLoss

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
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }
    
    # Define policy architecture
    policy_dim = 256

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Mish,
        out_features=num_outputs,  # predict only loc
        num_cells=[policy_dim, policy_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [policy_dim, policy_dim]],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    # Add probabilistic sampling of the actions
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
        default_interaction_type=InteractionType.RANDOM,
    )

    # Define value architecture
    value_dim = 256
    vmin = cfg.network.vmin
    vmax = cfg.network.vmax
    nbins = cfg.network.nbins
    dk = (vmax - vmin) / (nbins - 4)
    ktot = dk * nbins
    vmax_support = math.ceil(vmin + ktot)

    value_support = torch.linspace(vmin, vmax_support, nbins)

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Mish,
        out_features=nbins,
        num_cells=[value_dim, value_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                        "normalized_shape": hidden_size} for hidden_size in [value_dim, value_dim]],
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation_vector"],
    )
    support_network = SupportOperator(value_support)
    value_module_1 = TensorDictModule(
        module=value_mlp,
        in_keys=["observation_vector"],
        out_keys=["state_value_logits"]
    )
    value_module_2 = TensorDictModule(
        module=support_network,
        in_keys=["state_value_logits"],
        out_keys=["state_value"]
    )

    value_module = TensorDictSequential(
        value_module_1, 
        value_module_2
    )
    policy_module = policy_module.to(device)
    value_module = value_module.to(device)
    return policy_module, value_module, value_support


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
            AltitudeToScaleCode(in_keys=["alt", "target_alt"], out_keys=["alt_code", "target_alt_code"],
                                            add_cosine=False),
            AltitudeToScaleCode(in_keys=["u", "v", "w", "udot", "vdot", "wdot"], 
                                out_keys=["u_code", "v_code", "w_code", "udot_code", "vdot_code", "wdot_code"],
                                            add_cosine=False, base_scale=0.1),
            Difference(in_keys=["target_alt_code", "alt_code", "target_speed", "mach"], 
                       out_keys=["altitude_error", "speed_error"]),
            AngularDifference(in_keys=["target_heading", "psi"], out_keys=["heading_error"]),
            PlanarAngleCosSin(in_keys=["psi"], out_keys=["psi_cos_sin"]),
            EulerToRotation(in_keys=["psi", "theta", "phi"], out_keys=["rotation"]),
            CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "psi_cos_sin", "rotation", 
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

@hydra.main(version_base="1.1", config_path="configs", config_name="main_ppo")
def main(cfg: DictConfig):
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

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

    actor_module, value_module, value_support = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
    loss_module = ClipHGaussPPOLoss(
        actor_module,
        value_module,
        clip_epsilon=cfg.ppo.clip_epsilon,
        loss_critic_type=cfg.ppo.loss_critic_type,
        entropy_coef=cfg.ppo.entropy_coef,
        critic_coef=cfg.optim.critic_coef,
        normalize_advantage= True,
        support=value_support
    )

    adv_module = GAE(
        gamma=cfg.ppo.gamma,
        lmbda=cfg.ppo.gae_lambda,
        value_network=value_module,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor_module.parameters(), lr=cfg.optim.lr_policy, eps=1e-5)
    critic_optim = torch.optim.Adam(value_module.parameters(), lr=cfg.optim.lr_value, eps=1e-5)

    collected_frames = 0
    cfg_logger_save_interval = cfg.logger.save_interval
    cfg_logger_eval_interval = cfg.logger.eval_interval
    loaded_frames = 0

    load_model = False
    if load_model:
        model_dir="2024-12-07/10-39-45/"
        model_name = "training_snapshot_8040000"
        loaded_state = load_model_state(model_name, model_dir)

        critic_state = loaded_state['model_critic']
        actor_optim_state = loaded_state['actor_optimizer']
        critic_optim_state = loaded_state['critic_optimizer']
        collected_frames = loaded_state['collected_frames']['collected_frames']
        loaded_frames = collected_frames
        value_module.load_state_dict(critic_state)
        actor_optim.load_state_dict(actor_optim_state)
        critic_optim.load_state_dict(critic_optim_state)


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
   # cfg_max_grad_norm = cfg.optim.max_grad_norm
    num_network_updates = 0
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr_policy = cfg.optim.lr_policy
    cfg_optim_lr_value = cfg.optim.lr_value
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
  
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            for reward_key in reward_keys:
                log_info.update(
                    {
                        f"train/{reward_key}": data["next", "episode_" + reward_key][
                            data["next", "done"]
                        ].mean().item()
                    }
                )

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            # Compute GAE

            with torch.no_grad():
                data = adv_module(data)
            data_extend = data.reshape(-1)
            data_buffer.extend(data_extend)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)

                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr_policy * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr_value * alpha
                num_network_updates += 1
                
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()

                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]
                # Backward pass
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
              

         # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        # norms_mean = norms.apply(lambda x: x.float().mean(), batch_size=[])
        # for key, value in norms_mean.items():
        #     log_info.update({f"train/{key}": value.item()})

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
                        'model_actor': actor_module.state_dict(),
                        'model_critic': value_module.state_dict(),
                   #     'model_dynamics': dynamics_module.state_dict(),
                   #     'model_encoder': encoder_module.state_dict(),
                   #     'model_reward': reward_module.state_dict(),
                        'actor_optimizer': actor_optim.state_dict(),
                        'critic_optimizer': critic_optim.state_dict(),
                   #     'consistency_optimizer': consistency_optim.state_dict(),
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
    