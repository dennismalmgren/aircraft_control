import os
import random
import time

import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from control_env import JSBSimControlEnv, JSBSimControlEnvConfig
from torchrl.envs.utils import step_mdp
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.data.tensor_specs import TensorSpec, Composite
from jsbsim_interface import AircraftSimulatorConfig, AircraftJSBSimSimulator, AircraftJSBSimInitialConditions
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

)
from torchrl.envs import (
    ParallelEnv,
    EnvCreator
)

from tensordict.nn import AddStateIndependentNormalScale
from tensordict.nn import TensorDictModule
from torchrl.envs import ExplorationType
from torchrl.modules.distributions import TanhNormal
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDict

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
            

def make_models(cfg, observation_spec: TensorSpec, action_spec: TensorSpec, device: torch.device):
    input_shape = observation_spec.shape
    num_outputs = action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
    }

    policy_mlp = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[256, 256],
        #norm_class=torch.nn.LayerNorm,
        #norm_kwargs=[{"elementwise_affine": False,
        #             "normalized_shape": hidden_size} for hidden_size in cfg.network.policy_hidden_sizes],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
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

    value_net = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Tanh,
        out_features=1,  # predict only loc
        num_cells=[256, 256],
    )

    for layer in value_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(
        value_net,
        in_keys=["observation_vector"],
    )
    
    return policy_module, value_module

def make_raw_environment():
    env = JSBSimControlEnv()
    return env
def apply_env_transforms(env):
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=2000),
            CatTensors(in_keys=["u", "v", "w", "udot", "vdot", "wdot", "phi", "theta", "psi", "p", "q", "r", 
                                "pdot", "qdot", "rdot", "lat", "lon", "alt", "air_density", "speed_of_sound", 
                                "crosswind", "headwind", "airspeed", "groundspeed", "last_action"],
                                    out_key="observation_vector", del_keys=False),
            VecNorm(in_keys=["observation_vector"], decay=0.99999, eps=1e-2),
            CatFrames(N=10, dim=-1, in_keys=["observation_vector"]),
            RewardSum()
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


@hydra.main(version_base="1.1", config_path="configs", config_name="main")
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

    template_env = make_environment()
    policy_module, value_module = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=cfg.ppo.clip_epsilon,
        loss_critic_type=cfg.ppo.loss_critic_type,
        entropy_coef=cfg.ppo.entropy_coef,
    )
    
    adv_module = GAE(
        gamma=cfg.ppo.gamma,
        lmbda=cfg.ppo.gae_lambda,
        value_network=value_module,
    )

    actor_optim = torch.optim.AdamW(policy_module.parameters(), lr=cfg.optim.lr_policy, eps=cfg.optim.eps)
    critic_optim = torch.optim.AdamW(value_module.parameters(), lr=cfg.optim.lr_value, eps=cfg.optim.eps)

    frames_remaining = cfg.collector.total_frames
    frames_collected = 0
    # Create collector
    collector = SyncDataCollector(
        create_env_fn=env_maker(cfg),
        policy=policy_module,
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
    
    reward_keys = ["reward"]
    cfg_loss_ppo_epochs = cfg.ppo.epochs
    cfg_max_grad_norm = cfg.optim.max_grad_norm
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, ncols=0)
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
        frames_collected += frames_in_batch
        pbar.update(frames_in_batch)
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                  #  "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )
            for reward_key in reward_keys:
                log_info.update(
                    {
                        f"train/{reward_key}": data["next", "episode_" + reward_key][
                            data["next", "done"]
                        ].mean().item()
                    }
                )
        data = data.select("observation_vector", 
                           "action", 
                           "sample_log_prob",
                           ("next", "reward"), 
                           ("next", "terminated"), 
                           ("next","done"), 
                           ("next", "observation_vector"))
        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            # Compute GAE

            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)
            data_buffer.extend(data_reshape)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                actor_optim.zero_grad()
                critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy_module.parameters(), cfg_max_grad_norm)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(value_module.parameters(), cfg_max_grad_norm)
                actor_optim.step()
                critic_optim.step()
                norms[j, k] = TensorDict({
                    "actor_grad_norm": actor_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
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
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, frames_collected)  

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
    