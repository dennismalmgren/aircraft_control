# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

#import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from tensordict import TensorDict
from torch.distributions.normal import Normal
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
from torchrl.collectors import SyncDataCollector
from torchrl.record.loggers import generate_exp_name, get_logger
import hydra

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

from control_env import JSBSimControlEnv, JSBSimControlEnvConfig
from transforms.euler_to_rotation_transform import EulerToRotation
from transforms.euler_to_rotation_transform import EulerToRotation
from transforms.altitude_to_scale_code_transform import AltitudeToScaleCode
from transforms.altitude_to_digits_transform import AltitudeToDigits
from transforms.min_max_transform import TimeMinPool, TimeMaxPool
from transforms.episode_sum_transform import EpisodeSum
from transforms.difference_transform import Difference
from transforms.angular_difference_transform import AngularDifference
from transforms.planar_angle_cos_sin_transform import PlanarAngleCosSin

def make_raw_environment() -> EnvBase:
    env = JSBSimControlEnv()
    return env

def apply_env_transforms(env):
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=2000),
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

def make_environment() -> EnvBase:
    env = make_raw_environment()
    env = apply_env_transforms(env)
    return env

def env_maker(cfg):
    parallel_env = ParallelEnv(
        cfg.num_envs,
        EnvCreator(lambda cfg=cfg: make_raw_environment()),
        #serial_for_single=True,
        #device=torch.device("cuda:0")
    )
    parallel_env = apply_env_transforms(parallel_env)
    return parallel_env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


@hydra.main(version_base="1.1", config_path="configs", config_name="main_lean")
def main(cfg: "DictConfig"):
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = batch_size // cfg.num_minibatches
    cfg.batch_size = cfg.num_minibatches * cfg.minibatch_size
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size

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

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)
    torch.manual_seed(cfg.random.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    ####### Environment setup #######
    #envs = make_environment()
    #envs = env_maker(cfg)
    envs = make_raw_environment()
    envs = apply_env_transforms(envs)
#    envs = gym.vector.SyncVectorEnv(
#        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
#    )
    n_act = envs.action_spec.shape[-1]
    n_obs = envs.observation_spec["observation_vector"].shape[-1]
    #n_act = math.prod(envs.single_action_space.shape)
    #n_obs = math.prod(envs.single_observation_space.shape)
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())

    ####### Agent #######
    agent = Agent(n_obs, n_act, device=device)
    # Make a version of agent with detached params
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(cfg.optim.lr_policy, device=device),
        eps=1e-5,
        capturable=cfg.use_cudagraphs and not cfg.use_torch_compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value


    def gae(next_dones, next_observation, value, next_reward):
        next_done = next_dones[:, -1]
        next_value = get_value(next_observation[:, -1]).reshape(-1, 1)
        lastgaelam = 0
        nextnonterminals = (~next_dones).float().unbind(1)
        vals = value
        vals_unbind = vals.unbind(1)
        rewards = next_reward.unbind(1)

        advantages = []
        nextnonterminal = (~next_done).float()
        nextvalues = next_value
        for t in range(cfg.num_steps - 1, -1, -1):
            cur_val = vals_unbind[t]
            delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - cur_val
            advantages.append(delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam)
            lastgaelam = advantages[-1]
            nextnonterminal = nextnonterminals[t]
            nextvalues = cur_val
        advantages = torch.stack(list(reversed(advantages)), dim=1)
        returns = advantages + vals
        return advantages, returns


    # def rollout(td):
    # # ts = []     
    #     tds = []
    #     for step in range(cfg.num_steps):
    #         # ALGO LOGIC: action logic
    #         obs = td['observation_vector']
    #         action, logprob, _, value = policy(obs=obs)
    #         td['action'] = action
    #         td['value'] = value
    #         td['logprobs'] = logprob.unsqueeze(-1)
    #         # TRY NOT TO MODIFY: execute the game and log data.
    #         next_td, td = envs.step_and_maybe_reset(td.to("cpu"))
    #         tds.append(next_td)
    #         td = td.to("cuda:0")

    #     td_rollout = torch.stack(tds, dim=1).to("cuda:0")
    #     return td, td_rollout


    def update(observation_vector, action, logprobs, advantages, returns, value):
        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(observation_vector, action)
        newlogprob = newlogprob.unsqueeze(-1) 
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > cfg.ppo.clip_epsilon).float().mean()

        if cfg.ppo.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_epsilon, 1 + cfg.ppo.clip_epsilon)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        #newvalue = newvalue.view(-1)
        if cfg.ppo.clip_value_loss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = value + torch.clamp(
                newvalue - value,
                -cfg.ppo.clip_epsilon,
                cfg.ppo.clip_epsilon,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - cfg.ppo.entropy_coef * entropy_loss + v_loss

        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), cfg.optim.max_grad_norm)
        optimizer.step()

        return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


    update = tensordict.nn.TensorDictModule(
        update,
        in_keys=["observation_vector", "action", "logprobs", "advantages", "returns", "value"],
        out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
    )

    gae_m = tensordict.nn.TensorDictModule(
        gae,
        in_keys=[("next", "done"), ("next", "observation_vector"), "value", ("next", "reward")],
        out_keys=["advantages", "returns"],
    )

    gae = gae_m
    # Compile policy
    raw_policy = policy
    if cfg.use_torch_compile:
        #policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if cfg.use_cudagraphs:
        #policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    collectorModule = tensordict.nn.TensorDictModule(
                        module=raw_policy,
                        in_keys=["observation_vector"],
                        out_keys=["action", "logprobs", "entropy", "value"])
    
    collector = SyncDataCollector(
        create_env_fn=env_maker(cfg),
        policy=collectorModule,
        frames_per_batch=cfg.batch_size,
        total_frames=cfg.total_timesteps,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        compile_policy=cfg.use_torch_compile,
        cudagraph_policy=cfg.use_cudagraphs,
    )
    
    reward_keys = ["reward", "task_reward", "smoothness_reward"]

    global_step = 0
    #reset_td = envs.reset()
    #next_td = reset_td.to(device)
    pbar = tqdm.tqdm(total=cfg.total_timesteps)
    global_step_burnin = None
    collected_frames = 0
    torch.compiler.cudagraph_mark_step_begin()
    for i, data in enumerate(collector):
        log_info = {}
        # Annealing the rate if instructed to do so.
        if cfg.optim.anneal_lr:
            frac = 1.0 - i / cfg.num_iterations
            lrnow = frac * cfg.optim.lr_policy
            optimizer.param_groups[0]["lr"].copy_(lrnow)
        collected_frames += data.numel()

        pbar.update(data.numel())
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        episode_pdot = data["next", "episode_pdot"][data["next", "done"]]
        episode_p = data["next", "episode_p"][data["next", "done"]]
        episode_min_mach = data["next", "episode_min_mach"][data["next", "done"]]
        episode_max_mach = data["next", "episode_max_mach"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/pdot": (episode_pdot / episode_length).mean().item(),
                    "train/p": (episode_p / episode_length).mean().item(),
                    "train/episode_length": episode_length.float().mean().item(), #mean?
                    "train/episode_min_mach": episode_min_mach.mean().item(),
                    "train/episode_max_mach": episode_max_mach.mean().item(),
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

        for epoch in range(cfg.ppo.epochs):
            data = gae(data)
            container_flat = data.view(-1)

            b_inds = torch.randperm(container_flat.shape[0], device=device).split(cfg.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                
            else:
                continue
            break
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)  

        collector.update_policy_weights_()

        #if i > 1 and (i + 1) % 1 == 0:
            #speed = (global_step - global_step_burnin) / (time.time() - start_time)
           # r = container_flat["next", "reward"].mean()
            #r_max = container_flat["next", "reward"].max()
            #avg_returns_t = torch.tensor(avg_returns).mean()

    
          #  lr = optimizer.param_groups[0]["lr"]
           # pbar.set_description(
                #f"speed: {speed: 4.1f} sps, "
           #     f"reward avg: {r :4.2f}, "
           #     f"reward max: {r_max:4.2f}, "
               # f"returns: {avg_returns_t: 4.2f},"
                #f"lr: {lr: 4.2f}"
           # )

        torch.compiler.cudagraph_mark_step_begin()

    envs.close()

if __name__=="__main__":
    main()