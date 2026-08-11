import argparse
import os
import logging
import sys
import time
import numpy as np
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGGER_PATH = BASE_DIR / "logs" / "ppo"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from environments.sorl_lander import RocketLanding6DOFEnv

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(LOGGER_PATH / "ppo_training_log.log"), mode="w"),
        logging.StreamHandler(),
    ],
)

CKPT_DIR = BASE_DIR / "ppo_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_EVERY = 10
EVAL_EVERY = 10
EVAL_EPISODES = 20
KEEP_LAST = 20
N_ENVS = 8

ADVANCE_RATE = 0.75
ADVANCE_WINDOW = 20


def make_env(curriculum_level=0):
    def _init():
        return RocketLanding6DOFEnv(curriculum_level=curriculum_level)

    return _init


def save_checkpoint(path, policy, optimizer, update, extra=None):
    torch.save(
        {
            "update": update,
            "model_state": policy.state_dict(),
            "optim_state": optimizer.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def prune_rolling_checkpoints(ckpt_dir, keep_last):
    files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_upd")],
        key=lambda x: int(x.split("upd")[1].split(".")[0]),
    )
    for f in files[:-keep_last]:
        os.remove(os.path.join(ckpt_dir, f))


@torch.no_grad()
def deterministic_eval(policy, make_env_fn, n_episodes):
    env = make_env_fn()
    lands = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean_action = policy.actor_mean(obs_t).squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(mean_action)
            done = terminated or truncated
        if info.get("termination") == "LAND":
            lands += 1
    env.close()
    return lands / n_episodes


def _mlp(dims, activation=nn.Tanh):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PPONet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.actor_mean = _mlp([obs_dim] + hidden + [action_dim])
        self.actor_logstd = nn.Parameter(torch.ones(action_dim) * -1.0)
        self.critic = _mlp([obs_dim] + hidden + [1])

        for m in self.actor_mean.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        last_actor = [m for m in self.actor_mean.modules() if isinstance(m, nn.Linear)][
            -1
        ]
        nn.init.orthogonal_(last_actor.weight, gain=0.01)
        last_critic = [m for m in self.critic.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_critic.weight, gain=0.01)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs)
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(
        self,
        n_steps,
        n_envs,
        obs_dim,
        action_dim,
        gamma1,
        gamma2,
        device,
        gae_lambda=0.95,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.lam = gae_lambda
        self.device = device
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, action_dim, device=device)
        self.shaping_rewards = torch.zeros(n_steps, n_envs, device=device)
        self.terminal_rewards = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, r_shaping, r_terminal, value, done):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        self.log_probs[i] = torch.as_tensor(
            log_prob, dtype=torch.float32, device=self.device
        )
        self.shaping_rewards[i] = torch.as_tensor(
            r_shaping, dtype=torch.float32, device=self.device
        )
        self.terminal_rewards[i] = torch.as_tensor(
            r_terminal, dtype=torch.float32, device=self.device
        )
        self.values[i] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.dones[i] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def compute_advantages(self, last_value: torch.Tensor):
        advantages = torch.zeros(self.n_steps, self.n_envs, device=self.device)
        last_gae = torch.zeros(self.n_envs, device=self.device)
        lv = last_value.detach().squeeze(-1)
        rewards = self.shaping_rewards + self.terminal_rewards  # combine

        for t in reversed(range(self.n_steps)):
            mask = 1.0 - self.dones[t]
            delta = rewards[t] + self.gamma1 * lv * mask - self.values[t]
            last_gae = delta + self.gamma1 * self.lam * mask * last_gae
            advantages[t] = last_gae
            lv = self.values[t]

        returns = advantages + self.values
        return advantages, returns

    def reset(self):
        self.ptr = 0


class ProximalPolicyOptimisation:
    def __init__(
        self,
        device,
        total_timesteps=3_000_000,
        n_steps=512,
        resume_from=None,
        start_curriculum=0,
    ):
        self.total_timesteps = total_timesteps
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
        self.n_steps = n_steps
        self.resume_from = resume_from
        self.start_curriculum = start_curriculum
        self.curriculum_level = start_curriculum
        self.env = SubprocVecEnv(
            [make_env(self.curriculum_level) for _ in range(N_ENVS)]
        )

    def _rebuild_envs(self, curriculum_level):
        self.env.close()
        self.curriculum_level = curriculum_level
        self.env = SubprocVecEnv([make_env(curriculum_level) for _ in range(N_ENVS)])
        return self.env.reset()

    def train(
        self,
        hidden=None,
        n_epochs=6,
        batch_size=256,
        lr=1e-4,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        anneal_lr=False,
    ):
        TRAINING_LOG_PATH = LOGGER_PATH / "ppo_training_log.csv"
        CL_TRANSITIONS_PATH = LOGGER_PATH / "ppo_curriculum_transitions.csv"

        _training_log_fields = [
            "update_num",
            "total_steps",
            "curriculum_level",
            "fps",
            "mean_return",
            "std_return",
            "mean_ep_len",
            "land_rate",
            "crash_rate",
            "tip_rate",
            "oob_rate",
            "timeout_rate",
            "pg_loss",
            "vf_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "explained_var",
            "grad_norm",
            "logstd_T",
            "logstd_pitch",
            "logstd_lat",
            "eval_land_rate",
            "learning_rate",
        ]
        _cl_fields = [
            "update_num",
            "total_steps",
            "from_level",
            "to_level",
            "window_land_rate",
            "stable_eval",
        ]

        train_log_file = open(TRAINING_LOG_PATH, "w", newline="")
        cl_log_file = open(CL_TRANSITIONS_PATH, "w", newline="")
        csv_writer = csv.DictWriter(train_log_file, fieldnames=_training_log_fields)
        cl_writer = csv.DictWriter(cl_log_file, fieldnames=_cl_fields)
        csv_writer.writeheader()
        cl_writer.writeheader()

        if hidden is None:
            hidden = [256, 256]

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr0 = lr
        self.clip = clip_coef
        self.ent_c = ent_coef
        self.vf_c = vf_coef
        self.max_gnorm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.hidden = hidden
        self.gae_lambda = gae_lambda

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.net = PPONet(obs_dim, action_dim, self.hidden).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr0, eps=1e-5)

        if self.resume_from and os.path.isfile(self.resume_from):
            ckpt = torch.load(self.resume_from, map_location=self.device)
            self.net.load_state_dict(ckpt["model_state"])
            self.opt.load_state_dict(ckpt["optim_state"])
            logger.info(
                f"Resumed from {self.resume_from} (saved at update {ckpt['update']})"
            )

        self.buffer = RolloutBuffer(
            self.n_steps,
            N_ENVS,
            obs_dim,
            action_dim,
            0.99,
            0.99,
            self.device,
            gae_lambda,
        )

        raw_obs = self.env.reset()
        total_steps = 0
        update_num = 0
        n_updates = self.total_timesteps // self.n_steps

        ep_return_buf = []
        ep_len_buf = []
        termination_buf = []
        success_buf = []

        curriculum_level = self.start_curriculum
        advance_window_buf = []
        best_eval_land_rate = -1.0
        last_eval_rate = 0.0
        eval_history = []

        cur_ep_return = 0.0
        cur_ep_len = 0
        last_info_snapshot = {}

        t0 = time.time()

        current_eval_land_rate = float("nan")

        while total_steps < self.total_timesteps:
            update_num += 1

            if self.anneal_lr:
                frac = 1.0 - (update_num - 1) / max(n_updates, 1)
                for pg in self.opt.param_groups:
                    pg["lr"] = frac * self.lr0

            self.buffer.reset()
            rollout_returns = []
            rollout_lens = []
            rollout_terms = []

            for step in range(self.n_steps):
                obs_tensor = torch.from_numpy(np.asarray(raw_obs, dtype=np.float32)).to(
                    self.device
                )
                with torch.no_grad():
                    action, log_prob, _, value = self.net.get_action_and_value(
                        obs_tensor
                    )
                action_np = action.cpu().numpy()
                raw_next, reward, terminated, info = self.env.step(action_np)
                dones = terminated
                r_shaping = np.array([i["shaping_reward"] for i in info])
                r_terminal = np.array([i["terminal_reward"] for i in info])

                done = dones

                self.buffer.add(
                    obs_tensor,
                    action,
                    log_prob,
                    r_shaping,
                    r_terminal,
                    value.squeeze(),
                    done,
                )
                total_steps += N_ENVS

                cur_ep_return += float(reward.sum())
                cur_ep_len += 1
                last_info_snapshot = info

                raw_obs = raw_next

                for env_idx, d in enumerate(dones):
                    if d:
                        ep_info = info[env_idx]
                        rollout_returns.append(
                            ep_info.get("episode_return", cur_ep_return)
                        )
                        rollout_lens.append(ep_info.get("episode_length", cur_ep_len))
                        rollout_terms.append(ep_info.get("termination", "-"))
                        success_buf.append(
                            1 if ep_info.get("termination") == "LAND" else 0
                        )
                        cur_ep_return = 0.0
                        cur_ep_len = 0

            with torch.no_grad():
                last_val = self.net.get_value(
                    torch.from_numpy(np.asarray(raw_obs, dtype=np.float32)).to(
                        self.device
                    )
                )

            advantages, returns = self.buffer.compute_advantages(last_val)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pg_losses, vf_losses, ent_vals, kl_vals, clipfracs, grad_norms = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            idx = np.arange(self.n_steps * N_ENVS)
            b_obs = self.buffer.obs.reshape(-1, obs_dim)
            b_actions = self.buffer.actions.reshape(-1, action_dim)
            b_log_probs = self.buffer.log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            y_true = b_returns
            y_pred = self.buffer.values.reshape(-1)
            var_y = y_true.var()
            explained_var = float(1.0 - (y_true - y_pred).var() / (var_y + 1e-8))

            for _ in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, self.n_steps * N_ENVS, self.batch_size):
                    mb = torch.tensor(
                        idx[start : start + self.batch_size], device=self.device
                    )
                    _, new_lp, ent, new_val = self.net.get_action_and_value(
                        b_obs[mb], b_actions[mb]
                    )
                    logratio = new_lp - b_log_probs[mb]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        if approx_kl > 0.02:
                            break
                        clipfrac = (
                            ((ratio - 1.0).abs() > self.clip).float().mean().item()
                        )

                    adv_mb = b_advantages[mb]
                    pg_loss = torch.max(
                        -adv_mb * ratio,
                        -adv_mb * ratio.clamp(1 - self.clip, 1 + self.clip),
                    ).mean()
                    vf_loss = 0.5 * ((new_val.squeeze(-1) - b_returns[mb]) ** 2).mean()
                    entropy = ent.mean()
                    loss = pg_loss + self.vf_c * vf_loss - self.ent_c * entropy

                    self.opt.zero_grad()
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gnorm)
                    self.opt.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_vals.append(entropy.item())
                    kl_vals.append(approx_kl)
                    clipfracs.append(clipfrac)
                    grad_norms.append(float(gn))

            ep_return_buf.extend(rollout_returns)
            ep_len_buf.extend(rollout_lens)
            termination_buf.extend(rollout_terms)
            ep_return_buf = ep_return_buf[-100:]
            ep_len_buf = ep_len_buf[-100:]
            termination_buf = termination_buf[-100:]
            success_buf = success_buf[-100:]

            elapsed = time.time() - t0
            fps = total_steps / max(elapsed, 1e-6)
            mean_ret = np.mean(ep_return_buf) if ep_return_buf else float("nan")
            mean_len = np.mean(ep_len_buf) if ep_len_buf else float("nan")
            land_rate = 100.0 * (sum(success_buf) / max(len(success_buf), 1))
            n_eps_update = len(rollout_returns)

            term_counts = {}
            for t in termination_buf:
                term_counts[t] = term_counts.get(t, 0) + 1
            term_str = " ".join(f"{k}:{v}" for k, v in term_counts.items()) or "—"
            std_ret = float(np.std(ep_return_buf)) if ep_return_buf else float("nan")
            n_terms = max(len(termination_buf), 1)
            crash_rate = 100.0 * term_counts.get("CRASH", 0) / n_terms
            tip_rate = 100.0 * term_counts.get("TIP", 0) / n_terms
            oob_rate = 100.0 * term_counts.get("OOB", 0) / n_terms
            timeout_rate = 100.0 * term_counts.get("TIMEOUT", 0) / n_terms

            with torch.no_grad():
                logstd_mean = self.net.actor_logstd.mean().item()
                _logstd = self.net.actor_logstd.detach().cpu().numpy()
                logstd_T, logstd_pitch, logstd_lat = (
                    float(_logstd[0]),
                    float(_logstd[1]),
                    float(_logstd[2]),
                )

            info_s = info[0]

            logger.info(
                f"[Upd {update_num:>4}/{n_updates}] "
                f"steps={total_steps:>8} fps={fps:>5.0f} "
                f"ep_ret={mean_ret:>7.2f} ep_len={mean_len:>5.1f} n_ep={n_eps_update:>3} "
                f"land={land_rate:>4.1f}% "
                f"pg={np.mean(pg_losses):+.4f} vf={np.mean(vf_losses):6.3f} "
                f"ent={np.mean(ent_vals):+.3f} kl={np.mean(kl_vals):+.4f} "
                f"clip={np.mean(clipfracs):.3f} logstd={logstd_mean:+.2f} "
                f"lr={self.opt.param_groups[0]['lr']:.2e}"
                f"explained var={explained_var:+.3f}"
            )
            logger.info(
                f"           last_state: "
                f"z={info_s.get('z', float('nan')):6.2f} "
                f"vz={info_s.get('vz', float('nan')):+6.2f} "
                f"x={info_s.get('x', float('nan')):+5.2f} y={info_s.get('y', float('nan')):+5.2f} "
                f"vx={info_s.get('vx', float('nan')):+5.2f} vy={info_s.get('vy', float('nan')):+5.2f} "
                f"roll={info_s.get('roll', float('nan')):+5.2f} pitch={info_s.get('pitch', float('nan')):+5.2f} "
                f"tilt degree={info_s.get('tilt_deg', float('nan')):+5.2f}"
                f"T_base={info_s.get('T_base', float('nan')):6.1f} "
                f"| terms[{term_str}]"
            )

            csv_writer.writerow(
                {
                    "update_num": update_num,
                    "total_steps": total_steps,
                    "curriculum_level": curriculum_level,
                    "fps": round(fps, 1),
                    "mean_return": round(mean_ret, 3),
                    "std_return": round(std_ret, 3),
                    "mean_ep_len": round(mean_len, 1),
                    "land_rate": round(land_rate, 2),
                    "crash_rate": round(crash_rate, 2),
                    "tip_rate": round(tip_rate, 2),
                    "oob_rate": round(oob_rate, 2),
                    "timeout_rate": round(timeout_rate, 2),
                    "pg_loss": round(float(np.mean(pg_losses)), 5),
                    "vf_loss": round(float(np.mean(vf_losses)), 5),
                    "entropy": round(float(np.mean(ent_vals)), 5),
                    "approx_kl": round(float(np.mean(kl_vals)), 5),
                    "clip_frac": round(float(np.mean(clipfracs)), 4),
                    "explained_var": round(explained_var, 4),
                    "grad_norm": (
                        round(float(np.mean(grad_norms)), 4)
                        if grad_norms
                        else float("nan")
                    ),
                    "logstd_T": round(logstd_T, 4),
                    "logstd_pitch": round(logstd_pitch, 4),
                    "logstd_lat": round(logstd_lat, 4),
                    "eval_land_rate": current_eval_land_rate,
                    "learning_rate": self.opt.param_groups[0]["lr"],
                }
            )
            train_log_file.flush()

            if update_num % CKPT_EVERY == 0:
                ckpt_path = CKPT_DIR / f"ckpt_upd{update_num:06d}.pt"
                save_checkpoint(
                    ckpt_path,
                    self.net,
                    self.opt,
                    update_num,
                    {"curriculum_level": curriculum_level},
                )
                prune_rolling_checkpoints(CKPT_DIR, KEEP_LAST)

            advance_window_buf.append(land_rate)
            if len(advance_window_buf) > ADVANCE_WINDOW:
                advance_window_buf.pop(0)

            if update_num % EVAL_EVERY == 0:
                eval_rate = deterministic_eval(
                    self.net,
                    lambda: RocketLanding6DOFEnv(curriculum_level=curriculum_level),
                    EVAL_EPISODES,
                )
                last_eval_rate = eval_rate
                current_eval_land_rate = round(eval_rate * 100.0, 2)
                eval_history.append(eval_rate)
                eval_history = eval_history[-2:]

                logger.info(
                    f"  [EVAL] upd={update_num} cl={curriculum_level} "
                    f"det_land={eval_rate * 100:.1f}%  (best={best_eval_land_rate * 100:.1f}%)"
                )

                if eval_rate > best_eval_land_rate:
                    best_eval_land_rate = eval_rate
                    best_path = os.path.join(CKPT_DIR, "best_eval.pt")
                    save_checkpoint(
                        best_path,
                        self.net,
                        self.opt,
                        update_num,
                        {
                            "curriculum_level": curriculum_level,
                            "eval_land_rate": eval_rate,
                        },
                    )
                    logger.info(
                        f"  [EVAL] New best: {eval_rate * 100:.1f}% -> saved best_eval.pt"
                    )

            stable_eval = min(eval_history) if len(eval_history) == 2 else 0.0

            EVAL_GATE = (
                0.95
                if curriculum_level == 0
                else (0.85 if curriculum_level in [1, 2] else 0.80)
            )

            if (
                curriculum_level <= 3
                and len(advance_window_buf) == ADVANCE_WINDOW
                and np.mean(advance_window_buf) >= ADVANCE_RATE * 100.0
                and stable_eval >= EVAL_GATE
            ):
                cl_path = CKPT_DIR / f"best_cl{curriculum_level}.pt"
                save_checkpoint(
                    cl_path,
                    self.net,
                    self.opt,
                    update_num,
                    {
                        "curriculum_level": curriculum_level,
                        "window_land_rate": float(np.mean(advance_window_buf)),
                        "stable_eval": float(stable_eval),
                    },
                )

                if curriculum_level < 3:
                    cl_writer.writerow(
                        {
                            "update_num": update_num,
                            "total_steps": total_steps,
                            "from_level": curriculum_level,
                            "to_level": curriculum_level + 1,
                            "window_land_rate": round(
                                float(np.mean(advance_window_buf)), 3
                            ),
                            "stable_eval": round(float(stable_eval), 3),
                        }
                    )
                    cl_log_file.flush()
                    logger.info(
                        f"PROGRESSION: CL {curriculum_level} -> {curriculum_level + 1} "
                        f"(window={np.mean(advance_window_buf):.1f}%, stable_eval={stable_eval*100:.1f}%)"
                    )
                    curriculum_level += 1
                    advance_window_buf.clear()
                    last_eval_rate = 0.0
                    eval_history = []
                    best_eval_land_rate = -1.0
                    raw_obs = self._rebuild_envs(curriculum_level)
                else:
                    cl_writer.writerow(
                        {
                            "update_num": update_num,
                            "total_steps": total_steps,
                            "from_level": 3,
                            "to_level": "MASTERED",
                            "window_land_rate": round(
                                float(np.mean(advance_window_buf)), 3
                            ),
                            "stable_eval": round(float(stable_eval), 3),
                        }
                    )
                    cl_log_file.flush()
                    logger.info(
                        f"CL 3 MASTERED; Saved best_cl3.pt"
                        f"(window={np.mean(advance_window_buf):.1f}%, stable_eval={stable_eval*100:.1f}%)"
                    )
                    advance_window_buf.clear()

        train_log_file.close()
        cl_log_file.close()
        logger.info("Training logs written and closed.")


CURRICULUM_TRAIN = 0

if __name__ == "__main__":

    ppo = ProximalPolicyOptimisation(
        "cpu",
        10_000_000,
        512,
        resume_from=CKPT_DIR / f"{CURRICULUM_TRAIN - 1}",
        start_curriculum=CURRICULUM_TRAIN,
    )
    ppo.train()
