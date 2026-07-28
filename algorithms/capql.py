import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import sys
import torch
import wandb
import envs

from pathlib import Path

from morl_baselines.common.performance_indicators import hypervolume, sparsity
from morl_baselines.multi_policy.capql.capql import CAPQL
from mo_gymnasium.wrappers import MONormalizeReward
from morl_baselines.common.evaluation import log_all_multi_policy_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "weights"


if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CURRICULUM_LEVEL = 1

shared_curriculum = {"level": CURRICULUM_LEVEL}

env = mo_gym.make(
    "RocketBooster-capql-v0",
    curriculum_level=CURRICULUM_LEVEL,
    shared_curriculum=shared_curriculum,
)
env = gym.wrappers.NormalizeObservation(env)
env.unwrapped.allow_auto_promote = True

eval_env = mo_gym.make(
    "RocketBooster-capql-v0",
    curriculum_level=1,
    allow_auto_promote=False,
    shared_curriculum=shared_curriculum,
)
eval_env = gym.wrappers.NormalizeObservation(eval_env)
eval_env.unwrapped.allow_auto_promote = False
eval_env.obs_rms = env.obs_rms

wandb.init(
    project="RocketBooster_CAPQL",
    name="capql_curriculum2",
    config={"env_id": "RocketBooster-capql-v0", "timesteps": 750_000},
)

agent = CAPQL(
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    device="cpu",
    buffer_size=1000000,
    net_arch=[256, 256, 128],
    learning_starts=50000,
    gradient_updates=1,
    alpha=0.1,
    seed=42,
    experiment_name=f"capql_curriculum{CURRICULUM_LEVEL}",
    num_q_nets=3,
    log=True,
)

ref_point = np.array([-300.0, -300.0], dtype=np.float32)

agent.train(
    total_timesteps=750_000,
    eval_env=eval_env,
    ref_point=ref_point,
    num_eval_weights_for_front=50,
    num_eval_episodes_for_front=15,
    num_eval_weights_for_eval=20,
    eval_freq=10000,
    checkpoints=False,
    save_freq=100000,
)

torch.save(env.obs_rms, WEIGHTS_PATH / f"obs_rms_cl{CURRICULUM_LEVEL}.pt")
agent.save(save_dir=WEIGHTS_PATH, filename=f"capql_rocket_cl{CURRICULUM_LEVEL}")

try:
    eval_env.update_running_mean = False
except Exception:
    pass


def eval_weight(agent, env, w, n_episodes=30):
    w = np.asarray(w, dtype=np.float32)
    env.unwrapped.desired_weight = w
    counts = {"LAND": 0, "CRASH": 0, "TIP": 0, "OOB": 0, "TIMEOUT": 0}
    vecs, td = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep = np.zeros(2, dtype=np.float32)
        info, done = {}, False
        while not done:
            action = agent.eval(obs, w)
            obs, rvec, terminated, truncated, info = env.step(action)
            ep += np.asarray(rvec, dtype=np.float32)
            done = terminated or truncated
        vecs.append(ep)
        r = info.get("termination", "TIMEOUT")
        counts[r] = counts.get(r, 0) + 1
        td.append(
            [
                info.get("vz", np.nan),
                info.get("vx", np.nan),
                info.get("vy", np.nan),
                info.get("tilt_deg", np.nan),
                info.get("T_base", np.nan),
            ]
        )
    vecs, td, n = np.array(vecs), np.array(td), n_episodes
    print(
        f"w=[{w[0]:.2f},{w[1]:.2f}]  "
        f"LAND={100*counts['LAND']/n:3.0f}%  CRASH={100*counts['CRASH']/n:3.0f}%  "
        f"TIP={100*counts['TIP']/n:3.0f}%  |  "
        f"prec={vecs[:,0].mean():7.2f}  fuel={vecs[:,1].mean():7.2f}  |  "
        f"vz={np.nanmean(td[:,0]):5.2f}  |vh|={np.nanmean(np.abs(td[:,1:3])):4.2f}  "
        f"tilt={np.nanmean(td[:,3]):4.1f}deg  T={np.nanmean(td[:,4]):5.0f}"
    )
    return vecs.mean(axis=0)


weights = [
    np.array([a, 1.0 - a], np.float32)
    for a in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
]

front = np.array([eval_weight(agent, eval_env, w) for w in weights])

final_hv = hypervolume(ref_point=ref_point, points=front)
final_sparsity = sparsity(front)

print(f"Final Front - Hypervolume: {final_hv:.2f} | Sparsity: {final_sparsity:.2f}")

wandb.log(
    {
        "eval/final_manual_hypervolume": final_hv,
        "eval/final_manual_sparsity": final_sparsity,
    }
)

wandb.finish()
