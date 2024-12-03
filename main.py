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

class SoftmaxLayer(torch.nn.Module):
    def __init__(self,  internal_dim: int):
        super().__init__()
        self.internal_dim = internal_dim

    def forward(self, x):
        x_shape = x.shape
        x = x.view(*x_shape[:-1], self.internal_dim, -1)
        x = torch.softmax(x, dim=-1)
        return torch.clamp(x, self.vmin, self.vmax)
    
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
    layer_width = 256

    policy_mlp_1 = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.Mish,
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
            torch.nn.init.orthogonal_(layer.weight, 0.7)
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
        activation_class=torch.nn.Mish,
        out_features=num_outputs,  # predict only loc
        num_cells=[layer_width,layer_width],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                     "normalized_shape": hidden_size} for hidden_size in [layer_width,layer_width]],
    )
    # Initialize policy weights
    for layer in policy_mlp_2.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
    clamp_operator = ClampOperator(-10, 10)

    policy_mlp = torch.nn.Sequential(
        policy_mlp_1,
        policy_mlp_2,
        clamp_operator,
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

    if cfg.ppo.loss_critic_type == "hgauss":
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
            StepCounter(max_steps=2000),
            TimeMinPool(in_keys="mach", out_keys="episode_min_mach", T=2000),
            TimeMaxPool(in_keys="mach", out_keys="episode_max_mach", T=2000),
            RewardScaling(loc=0.0, scale=0.01, in_keys=["u", "v", "w", "udot", "vdot", "wdot"]),
            EulerToRotation(in_keys=["psi", "theta", "phi"], out_keys=["rotation"]),
            AltitudeToScaleCode(in_keys=["alt", "target_alt"], out_keys=["alt_code", "target_alt_code"], add_cosine=False),
            Difference(in_keys=["target_alt_code", "alt_code", "target_speed", "mach"], out_keys=["altitude_error", "speed_error"]),
            PlanarAngleCosSin(in_keys=["psi"], out_keys=["psi_cos_sin"]),
            AngularDifference(in_keys=["target_heading", "psi"], out_keys=["heading_error"]),                        
                    
            CatTensors(in_keys=["altitude_error", "speed_error", "heading_error", "alt_code", "mach", "psi_cos_sin", "rotation", "u", "v", "w", "udot", "vdot", "wdot",
                                "p", "q", "r", "pdot", "qdot", "rdot", "last_action"],
                                    out_key="observation_vector", del_keys=False),        
            CatFrames(N=10, dim=-1, in_keys=["observation_vector"]),
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

    template_env = make_environment()
    #test_td = template_env.reset()
    if cfg.ppo.loss_critic_type == "l2":
        policy_module, value_module, policy_parameters = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
        loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=cfg.ppo.clip_epsilon,
            loss_critic_type=cfg.ppo.loss_critic_type,
            entropy_coef=cfg.ppo.entropy_coef,
            normalize_advantage= True
        )
    elif cfg.ppo.loss_critic_type == "hgauss":
        policy_module, value_module, support = make_models(cfg, template_env.observation_spec["observation_vector"], template_env.action_spec, device)
        loss_module = ClipHGaussPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=cfg.ppo.clip_epsilon,
            loss_critic_type=cfg.ppo.loss_critic_type,
            entropy_coef=cfg.ppo.entropy_coef,
            normalize_advantage= False,
            support=support
        )

    adv_module = GAE(
        gamma=cfg.ppo.gamma,
        lmbda=cfg.ppo.gae_lambda,
        value_network=value_module,
    )
    #if use_torch_compile:
    #    torch.set_float32_matmul_precision('high')
    #    loss_module = torch.compile(loss_module)
        #adv_module = torch.compile(adv_module)
        #policy_module = torch.compile(policy_module)
        #value_module = torch.compile(value_module)

    actor_optim = torch.optim.AdamW(policy_module.parameters(), lr=cfg.optim.lr_policy, eps=cfg.optim.eps)
    critic_optim = torch.optim.AdamW(value_module.parameters(), lr=cfg.optim.lr_value, eps=cfg.optim.eps)

    collected_frames = 0
    cfg_logger_save_interval = cfg.logger.save_interval
    loaded_frames = 0

    load_model = False
    if load_model:
        model_dir="2024-11-27/02-22-57/"
        model_name = "training_snapshot_39040000"
        loaded_state = load_model_state(model_name, model_dir)

        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        actor_optim_state = loaded_state['actor_optimizer']
        critic_optim_state = loaded_state['critic_optimizer']
        collected_frames = loaded_state['collected_frames']['collected_frames']
        loaded_frames = collected_frames
        policy_module.load_state_dict(actor_state)
        value_module.load_state_dict(critic_state)
        actor_optim.load_state_dict(actor_optim_state)
        critic_optim.load_state_dict(critic_optim_state)


    frames_remaining = cfg.collector.total_frames - collected_frames

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=env_maker(cfg),
        policy=policy_module,
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
    
    
    reward_keys = ["reward", "task_reward", "smoothness_reward"]
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

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)
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
            data_extend = data.reshape(-1)
            data_buffer.extend(data_extend)
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
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy_module.parameters(), cfg_max_grad_norm)
                actor_optim.step()
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(value_module.parameters(), cfg_max_grad_norm)
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

        if ((i - 1) * frames_in_batch + loaded_frames) // cfg_logger_save_interval < \
           (i * frames_in_batch + loaded_frames) // cfg_logger_save_interval:
                savestate = {
                        'model_actor': policy_module.state_dict(),
                        'model_critic': value_module.state_dict(),
                        'actor_optimizer': actor_optim.state_dict(),
                        'critic_optimizer': critic_optim.state_dict(),
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
    