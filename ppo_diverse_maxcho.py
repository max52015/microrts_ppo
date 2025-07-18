import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder


def set_environment():
    run = None
    CHECKPOINT_FREQUENCY = 50

    # Argument parsing
    parser = argparse.ArgumentParser(description="PPO agent")
    # Common args
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py")
    )
    parser.add_argument("--gym-id", type=str, default="MicrortsDefeatCoacAIShaped-v3")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000000)
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True
    )
    parser.add_argument(
        "--prod-mode",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    # Algorithm-specific
    parser.add_argument("--n-minibatch", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--kle-stop",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--kle-rollback",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.n_minibatch

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print(f"Using device: {device}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment
    envs = MicroRTSVecEnv(
        num_envs=args.num_envs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_envs)],
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    envs = VecPyTorch(envs, device)
    if args.capture_video:
        experiment_name = (
            f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        envs = VecVideoRecorder(
            envs,
            f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 1000000 == 0,
            video_length=2000,
        )

    # Writer
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Optional wandb
    if not args.prod_mode:
        print("WandB without prod mode, if want plz use --prod-mode")
    else:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    return args, device, envs, writer, experiment_name, run, CHECKPOINT_FREQUENCY


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {
                    "r": ret,
                    "l": eplen,
                    "t": round(time.time() - self.tstart, 6),
                }
                info["episode"] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info["microrts_stats"] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, *, logits, masks):
        super().__init__(
            logits=torch.where(masks, logits, torch.tensor(-1e8, device=logits.device))
        )

    def entropy(self):
        # optional: masked entropy
        return super().entropy()


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class MicrortsUtils:
    def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def index_to_coord(idx, map_size=16):
        return idx % map_size, idx // map_size

    def predict_destination(pos, action, move_actions=[0, 1, 2, 3]):
        directions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        if action in move_actions:
            dx, dy = directions[action]
            return (pos[0] + dx, pos[1] + dy)
        return pos


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(self, envs, device):
        super().__init__()
        self.envs = envs
        self.device = device
        # Backbone：三層 ConvSequence
        c, h, w = envs.observation_space.shape[::-1]
        shape = (c, h, w)
        convs = []
        for oc in (16, 32, 32):
            seq = ConvSequence(shape, oc)
            shape = seq.get_output_shape()
            convs.append(seq)
        convs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], 256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*convs)
        self.actor = MicrortsUtils.layer_init(
            nn.Linear(256, envs.action_space.nvec.sum()), 0.01
        )
        self.critic = MicrortsUtils.layer_init(nn.Linear(256, 1), 1.0)

    # ---------------------------------------------------------
    def forward(self, obs):
        # obs [B,H,W,C] → [B,C,H,W]
        return self.network(obs.permute(0, 3, 1, 2).to(self.device))

    def get_value(self, obs):
        return self.critic(self.forward(obs))

    # ---------------------------------------------------------
    def _get_source_mask(self, batch):
        loc_np = np.asarray(self.envs.vec_client.getUnitLocationMasks(), dtype=np.bool_)
        loc_np = loc_np.reshape(batch, -1)
        return torch.as_tensor(loc_np, dtype=torch.bool, device=self.device)

    def _get_action_mask(self, chosen_units):
        mask_np = np.asarray(
            self.envs.vec_client.getUnitActionMasks(chosen_units), dtype=np.bool_
        )
        return torch.as_tensor(
            mask_np.reshape(len(chosen_units), -1), dtype=torch.bool, device=self.device
        )

    # ---------------------------------------------------------
    def get_action(self, obs, action=None, invalid_action_masks=None):
        B = obs.size(0)
        feats = self.forward(obs)
        logits = self.actor(feats)
        splits = self.envs.action_space.nvec.tolist()
        split_logits = torch.split(logits, splits, dim=1)

        # -------- sample mode --------
        if action is None:
            source_mask = self._get_source_mask(B)
            
            if source_mask.sum().item() == 0:
                raise RuntimeError("source_mask all False! Invalid action mask!")

            src_cat = CategoricalMasked(logits=split_logits[0], masks=source_mask)
            src_act = src_cat.sample()

            # parameter masks for each env
            param_mask = self._get_action_mask(src_act.cpu().numpy())
            if param_mask.sum().item() == 0:
                raise RuntimeError("param_mask all False! Invalid action mask!")
            
            split_param_masks = torch.split(param_mask, splits[1:], dim=1)

            acts = [src_act]
            for lg, msk in zip(split_logits[1:], split_param_masks):
                acts.append(CategoricalMasked(logits=lg, masks=msk).sample())
            action = torch.stack(acts)
            invalid_action_masks = torch.cat([source_mask, param_mask], dim=1)
        # -------- eval / update path --------
        cats = [
            CategoricalMasked(logits=lg, masks=msk.bool())
            for lg, msk in zip(split_logits, invalid_action_masks.split(splits, 1))
        ]
        logprob = torch.stack([c.log_prob(a) for c, a in zip(cats, action)]).sum(0)
        entropy = torch.stack([c.entropy() for c in cats]).sum(0)
        return action, logprob, entropy, invalid_action_masks


def main():
    # 初始化環境與參數
    args, device, envs, writer, experiment_name, run, CHECKPOINT_FREQUENCY = (
        set_environment()
    )
    agent = Agent(envs, device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        lr = lambda f: f * args.learning_rate

    # ALGO Logic: Storage for epoch data
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.observation_space.shape
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = torch.zeros(
        (args.num_steps, args.num_envs) + (envs.action_space.nvec.sum(),)
    ).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    if args.prod_mode and wandb.run.resumed:
        starting_update = run.summary.get("charts/update") + 1
        global_step = starting_update * args.batch_size
        api = wandb.Api()
        run = api.run(f"{run.entity}/{run.project}/{run.id}")
        model = run.file("agent.pt")
        model.download(f"models/{experiment_name}/")
        agent.load_state_dict(
            torch.load(f"models/{experiment_name}/agent.pt", map_location=device)
        )
        agent.eval()
        print(f"resumed at update {starting_update}")

    early_stop_K = 10  # 看過去多少次 update
    early_stop_max_var = 0.05  # 最小進步幅度
    recent_rewards = []  # 存每個update的平均reward

    for update in range(starting_update, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            # print(f"[DEBUG] get_value = {agent.get_value}")
            # envs.render()
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                action, logproba, _, invalid_action_masks[step] = agent.get_action(
                    obs[step]
                )

            actions[step] = action.T
            logprobs[step] = logproba

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rs, ds, infos = envs.step(action.T)
            rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

            for info in infos:
                if "episode" in info.keys():
                    print(
                        f"global_step={global_step}, episode_reward={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episode_reward", info["episode"]["r"], global_step
                    )
                    for key in info["microrts_stats"]:
                        writer.add_scalar(
                            f"charts/episode_reward/{key}",
                            info["microrts_stats"][key],
                            global_step,
                        )
                    break

        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape(
            (-1, invalid_action_masks.shape[-1])
        )

        # Optimizaing the policy and value network
        target_agent = Agent(envs, device).to(device)

        inds = np.arange(
            args.batch_size,
        )
        for i_epoch_pi in range(args.update_epochs):
            np.random.shuffle(inds)
            target_agent.load_state_dict(agent.state_dict())
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                _, newlogproba, entropy, _ = agent.get_action(
                    b_obs[minibatch_ind],
                    b_actions.long()[minibatch_ind].T,
                    b_invalid_action_masks[minibatch_ind],
                )
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                    v_clipped = b_values[minibatch_ind] + torch.clamp(
                        new_values - b_values[minibatch_ind],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.kle_stop:
                if approx_kl > args.target_kl:
                    break
            if args.kle_rollback:
                if (
                    b_logprobs[minibatch_ind]
                    - agent.get_action(
                        b_obs[minibatch_ind],
                        b_actions.long()[minibatch_ind].T,
                        b_invalid_action_masks[minibatch_ind],
                        envs,
                    )[1]
                ).mean() > args.target_kl:
                    agent.load_state_dict(target_agent.state_dict())
                    break

        # early stop 判斷
        avg_reward = float(rewards.mean().cpu().item())
        recent_rewards.append(avg_reward)
        if len(recent_rewards) > early_stop_K:
            recent_rewards.pop(0)  # 保持長度 = K
        if len(recent_rewards) == early_stop_K:
            mean_r = np.mean(recent_rewards)
            std_r = np.std(recent_rewards)
            if std_r / (mean_r + 1e-8) < early_stop_max_var:
                print(f"Early stopping triggered at update {update}!")
                if args.prod_mode:
                    torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                break

        ## CRASH AND RESUME LOGIC:
        if args.prod_mode:
            if not os.path.exists(f"models/{experiment_name}"):
                os.makedirs(f"models/{experiment_name}")
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"agent.pt")
            else:
                if update % CHECKPOINT_FREQUENCY == 0:
                    torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("charts/update", update, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar(
            "charts/sps", int(global_step / (time.time() - start_time)), global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
    print("Training completed successfully.")
    if wandb.run is not None:
        wandb.finish()
    print("WandB run finished.")
