import os
import numpy as np
import sys

from collections import deque, Counter
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "sac_ckpts"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from environments.sorl_lander import RocketLanding6DOFEnv

STEPS_PER_LEVEL = {0: 500_000, 1: 750_000, 2: 1_000_000, 3: 2_000_000}

os.makedirs(CHECKPOINT_PATH, exist_ok=True)


# EDIT BELOW CURRICULUM LEVEL
LEVEL = 0
if LEVEL - 1 > 0:
    RESUME_FROM = CHECKPOINT_PATH / f"/best_level{LEVEL-1}/best_model"
else:
    RESUME_FROM = None


def make_env(level, stats_path=None):
    env = DummyVecEnv([lambda: Monitor(RocketLanding6DOFEnv(curriculum_level=level))])
    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


class EpRewStdCallback(BaseCallback):
    """
    Class to log standard deviation for return std plots
    """

    def __init__(self, log_every=2000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            buf = self.model.ep_info_buffer
            if buf and len(buf) > 1:
                rewards = [ep["r"] for ep in buf]
                self.logger.record("rollout/ep_rew_std", float(np.std(rewards)))
        return True


class RocketMetricsCallback(BaseCallback):

    def __init__(self, window=100, log_every=2000, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.log_every = log_every
        self.terms = deque(maxlen=window)
        self.landed = deque(maxlen=window)
        self.last_state = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            term = info.get("termination", "")
            if term in ("LAND", "CRASH", "TIP", "OOB", "TIMEOUT"):
                self.terms.append(term)
                self.landed.append(1.0 if term == "LAND" else 0.0)
                self.last_state = {
                    k: info.get(k, float("nan"))
                    for k in ("z", "vz", "x", "y", "vx", "vy", "tilt_deg", "T_base")
                }

        if self.num_timesteps % self.log_every == 0 and self.terms:
            n = len(self.terms)
            counts = Counter(self.terms)
            land_rate = 100.0 * sum(self.landed) / len(self.landed)
            self.logger.record("rocket/land_rate", land_rate)
            for reason in ("CRASH", "TIP", "OOB", "TIMEOUT"):
                self.logger.record(
                    f"rocket/{reason.lower()}_rate", 100.0 * counts.get(reason, 0) / n
                )
            for k, v in self.last_state.items():
                self.logger.record(f"rocket_state/{k}", v)
        return True


# env: inherit previous level's normalization stats if resuming
prev_stats = CHECKPOINT_PATH / f"/vecnorm_level{LEVEL-1}.pkl" if LEVEL > 0 else None
env = make_env(LEVEL, stats_path=prev_stats)

# Increase learning_starts linearly with curriculum level
if RESUME_FROM:
    model = SAC.load(RESUME_FROM, env=env, device="cpu")
else:
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=25_000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        verbose=1,
    )

log_dir = f"{CHECKPOINT_PATH}/logs_level{LEVEL}"
os.makedirs(log_dir, exist_ok=True)
model.set_logger(configure(log_dir, ["stdout", "csv"]))

eval_env = make_env(LEVEL, stats_path=prev_stats)
eval_cb = EvalCallback(
    eval_env,
    eval_freq=10_000,
    n_eval_episodes=20,
    best_model_save_path=CHECKPOINT_PATH / f"/best_level{LEVEL}",
    deterministic=True,
)
ckpt_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=CHECKPOINT_PATH / f"/rolling_level{LEVEL}",
    save_replay_buffer=False,
)
rocket_cb = RocketMetricsCallback(window=100, log_every=2000)
rew_std_cb = EpRewStdCallback()
callbacks = CallbackList([eval_cb, rew_std_cb, ckpt_cb, rocket_cb])

try:
    model.learn(
        total_timesteps=10_000_000,
        reset_num_timesteps=True,
        callback=callbacks,
    )
except KeyboardInterrupt:
    print("Interrupted, saving current state.")
finally:
    model.save(CHECKPOINT_PATH / f"sac_level{LEVEL}")
    env.save(CHECKPOINT_PATH / f"/vecnorm_level{LEVEL}.pkl")
    print(f"Saved model + vecnorm for level {LEVEL}.")
