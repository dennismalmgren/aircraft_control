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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 20_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 20
    """the number of parallel game environments"""
    num_steps: int = 2000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 5
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 2
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

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
       # device=torch.device("cuda:0")
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

def gae2(next_dones, next_observation, value, next_reward):
    next_done = next_dones[-1]
    next_value = get_value(next_observation[-1]).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~next_dones).float().unbind(0)
    vals = value
    vals_unbind = vals.unbind(0)
    rewards = next_reward.unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]
        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val
    advantages = torch.stack(list(reversed(advantages)))
    returns = advantages + vals
    return advantages, returns

def gae(td_rollout):
    next_done = td_rollout['next', 'done'][-1]
    # bootstrap value if not done
    next_value = get_value(td_rollout['next', 'observation_vector'][-1]).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~td_rollout['next', "done"]).float().unbind(0)
    vals = td_rollout["value"]
    vals_unbind = vals.unbind(0)
    rewards = td_rollout["next", "reward"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val
    advantages = td_rollout["advantages"] = torch.stack(list(reversed(advantages)))

#    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
#    container["returns"] = advantages + vals
    td_rollout["returns"] = advantages + vals
    return td_rollout


def rollout(td):
   # ts = []
    tds = []
    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        td = td.to("cuda:0")
        obs = td['observation_vector']
        action, logprob, _, value = policy(obs=obs)
        td['action'] = action
        td['value'] = value
        td['logprobs'] = logprob.unsqueeze(-1)
        td_in = td.to("cpu")
        # TRY NOT TO MODIFY: execute the game and log data.
        next_td, td = envs.step_and_maybe_reset(td_in)
        tds.append(next_td) #TODO: args

    td_rollout = torch.stack(tds).to("cuda:0")
    return td.to("cuda:0"), td_rollout


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
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    #newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = value + torch.clamp(
            newvalue - value,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["observation_vector", "action", "logprobs", "advantages", "returns", "value"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

gae_m = tensordict.nn.TensorDictModule(
    gae2,
    in_keys=[("next", "done"), ("next", "observation_vector"), "value", ("next", "reward")
    out_keys=["advantages", "returns"],
)

if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    #envs = make_environment()
    envs = env_maker(args)

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
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        #gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    #avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    reset_td = envs.reset()
    #obs = reset_td["observation_vector"]
    next_td = reset_td.to(device)
    #next_obs = obs.to(device)
    #next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    # max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    # desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_input_td, td_rollout = rollout(next_td)

        td_rollout = gae(td_rollout)
        container_flat = td_rollout.view(-1)
        global_step += len(container_flat)
        next_td = next_input_td
        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        if global_step_burnin is not None and iteration % 1 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            r = td_rollout["next", "reward"].mean()
            r_max = td_rollout["next", "reward"].max()
            #avg_returns_t = torch.tensor(avg_returns).mean()

            with torch.no_grad():
                logs = {
                   # "episode_return": np.array(avg_returns).mean(),
                   # "logprobs": td_rollout["logprobs"].mean(),
                   # "advantages": td_rollout["advantages"].mean(),
                   # "returns": td_rollout["returns"].mean(),
                   # "vals": td_rollout["value"].mean(),
                   # "gn": out["gn"].mean(),
                }

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
               # f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )
            wandb.log(
                {"speed": speed, "r": r, "r_max": r_max, "lr": lr, **logs}, step=global_step
            )

    envs.close()