import os
import time
import logging
import random
import math
import sys
import torch

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from pathlib import Path
from gymnasium import spaces
from collections import deque

os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(10)

BASE_DIR = Path(__file__).resolve().parent.parent
LOGGER_PATH = BASE_DIR / "logs" / "morl"
MODEL_PATH = BASE_DIR / "models" / "rocket_model.urdf"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RUN_NAME = f"capql_{time.strftime('%Y%m%d_%H%M%S')}"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(LOGGER_PATH) / f"{RUN_NAME}.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)


class RocketLanding6DOFEnv(gym.Env):
    GRAVITY = 9.80
    MIN_THRUST, MAX_THRUST = (0.0, 300.0)
    MAX_GIMBAL_DEG = 12.0

    SIM_DT = 1.0 / 240.0
    N_SUBSTEPS = 12
    MAX_STEPS = 400

    BASE_HALF_LEN = 1.5
    TOUCHDOWN_Z = BASE_HALF_LEN + 0.05

    # landing gates
    LAND_VZ_MAX = 2.0
    LAND_VHORIZ_MAX = 1.0
    LAND_TILT_MAX = 0.3

    # safety / bounds
    TILT_WARN = 0.45
    TILT_LIMIT = 0.30
    MAX_X = 10.0
    MAX_Y = 10.0
    MAX_Z = 15.0

    # positive limits
    TILT_GOOD_LIMIT = 0.11

    # velocity-field guidance
    TAU = 1.2
    V0 = 5.0

    # reward terminals
    LANDING_BONUS = 100.0
    LAND_BASE_BONUS = 40.0
    LAND_QUALITY_BONUS = 60.0
    CRASH_PENALTY = -100.0
    OOB_PENALTY = -100.0
    TIP_PENALTY = -100.0

    ALT_SWITCH = 3.0
    TAU_HIGH = 1.2
    TAU_LOW = 0.6
    LAND_RADIUS_MAX = 1.0

    V_LOW = 0.8
    K_HORIZ = 0.5

    FUEL_DEPLETION_CONSTANT = 0.05
    FUEL_COST_SCALE = 3.0

    @property
    def curriculum_level(self):
        return self._shared_curriculum["level"]

    @curriculum_level.setter
    def curriculum_level(self, value):
        self._shared_curriculum["level"] = int(value)

    def __init__(
        self,
        render_mode=None,
        curriculum_level=0,
        allow_auto_promote=True,
        shared_curriculum=None,
    ):
        self.urdf_path = str(MODEL_PATH)
        self.render_mode = render_mode
        self.allow_auto_promote = allow_auto_promote

        if shared_curriculum is not None:
            self._shared_curriculum = shared_curriculum
        else:
            self._shared_curriculum = {"level": curriculum_level}

        super().__init__()

        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=50,
                cameraPitch=0,
                cameraTargetPosition=[5, -5, 3],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.GRAVITY)
        self.planeId = p.loadURDF("plane.urdf")
        self.rocketId = p.loadURDF(
            self.urdf_path,
            useFixedBase=False,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )

        for i in range(p.getNumJoints(self.rocketId)):
            if p.getJointInfo(self.rocketId, i)[1].decode() == "thrust_joint":
                self.thrust_link_index = i
                break
        self.mass = p.getDynamicsInfo(self.rocketId, -1)[0]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.reward_dim = 2
        self.reward_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            shape=(self.reward_dim,),
            dtype=np.float32,
        )
        self.steps = 0
        self.episode_count = 0
        self.LOG_EVERY = 100
        self._log_buffer = []
        self.total_steps = 0

        self._ep_return = 0.0
        self._ep_prec = 0.0
        self._ep_fuel = 0.0
        self._ret_window = deque(maxlen=100)
        self._land_window = deque(maxlen=100)

        self.MAX_CURRICULUM_LEVEL = 2
        self.PROMOTE_LAND_PCT = 0.85
        self.PROMOTE_MIN_EPISODES = 300
        self.PROMOTE_STABLE_WINDOWS = 3
        self.PROMOTE_STABILITY_BAND = 0.10
        self._episodes_at_level = 0
        self._land_rate_history = deque(maxlen=self.PROMOTE_STABLE_WINDOWS)

        self.current_pref = None

        self.reset()

        logging.info(
            f"Initialisation: auto_promote={allow_auto_promote} shared_passed={shared_curriculum is not None} level={self._shared_curriculum['level']}"
        )

    def _null_roll_rate(self):
        lin_v, ang_v = p.getBaseVelocity(self.rocketId)
        _, orn = p.getBasePositionAndOrientation(self.rocketId)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        w_body = R.T @ np.array(ang_v)
        w_body[2] = 0.0
        p.resetBaseVelocity(self.rocketId, lin_v, (R @ w_body).tolist())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode_count += 1
        self._episodes_at_level += 1

        if hasattr(self, "desired_weight") and self.desired_weight is not None:
            pref = np.asarray(self.desired_weight).flatten()
            if pref.size == 2:
                self._ep_pref = pref.astype(np.float32)
            else:
                self._ep_pref = np.array([1.0, 0.0], dtype=np.float32)
        else:
            self._ep_pref = np.array([1.0, 0.0], dtype=np.float32)

        cl = self.curriculum_level
        if cl == 0:
            startPos = [
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(1.7, 3.0),
            ]
            startEuler = [0.0, random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]
            startVel = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-1.0, -0.3),
            ]
        elif cl == 1:
            startPos = [
                random.uniform(-1.2, 1.2),
                random.uniform(-1.2, 1.2),
                random.uniform(2.5, 4.0),
            ]
            startEuler = [0.0, random.uniform(-0.07, 0.07), random.uniform(-0.07, 0.07)]
            startVel = [
                random.uniform(-0.8, 0.8),
                random.uniform(-0.8, 0.8),
                random.uniform(-1.5, -0.4),
            ]
        elif cl == 2:
            startPos = [
                random.uniform(-1.5, 1.5),
                random.uniform(-1.5, 1.5),
                random.uniform(3.0, 5.0),
            ]
            startEuler = [0.0, random.uniform(-0.09, 0.09), random.uniform(-0.09, 0.09)]
            startVel = [
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-2.0, -0.6),
            ]
        else:
            startPos = [
                random.uniform(-3.0, 3.0),
                random.uniform(-3.0, 3.0),
                random.uniform(8.0, 12.0),
            ]
            startEuler = [0.0, random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)]
            startVel = [
                random.uniform(-4, 4),
                random.uniform(-4, 4),
                random.uniform(-7.0, -3.0),
            ]

        startOrn = p.getQuaternionFromEuler(startEuler)
        p.resetBasePositionAndOrientation(self.rocketId, startPos, startOrn)
        p.resetBaseVelocity(self.rocketId, startVel, [0, 0, 0])
        self._null_roll_rate()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1

        action = np.asarray(action, dtype=np.float32)
        T_mag = (action[0] + 1.0) / 2.0 * self.MAX_THRUST
        T_mag = float(np.clip(T_mag, self.MIN_THRUST, self.MAX_THRUST))
        delta_pitch = np.radians(action[1] * self.MAX_GIMBAL_DEG)
        delta_lat = np.radians(action[2] * self.MAX_GIMBAL_DEG)

        Ry = np.array(
            [
                [np.cos(delta_pitch), 0, np.sin(delta_pitch)],
                [0, 1, 0],
                [-np.sin(delta_pitch), 0, np.cos(delta_pitch)],
            ]
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(delta_lat), -np.sin(delta_lat)],
                [0, np.sin(delta_lat), np.cos(delta_lat)],
            ]
        )
        F_body = T_mag * ((Ry @ Rx) @ np.array([0.0, 0.0, 1.0]))

        for _ in range(self.N_SUBSTEPS):
            _, orn = p.getBasePositionAndOrientation(self.rocketId)
            R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            F_world = R @ F_body
            thrust_pos_world = p.getLinkState(self.rocketId, self.thrust_link_index)[0]
            p.applyExternalForce(
                self.rocketId,
                -1,
                forceObj=F_world.tolist(),
                posObj=thrust_pos_world,
                flags=p.WORLD_FRAME,
            )
            p.stepSimulation()
            self._null_roll_rate()
            if self.render_mode == "human":
                time.sleep(self.SIM_DT)

        pos, orn = p.getBasePositionAndOrientation(self.rocketId)
        lin_vel, ang_vel = p.getBaseVelocity(self.rocketId)
        x, y, z = pos
        v = np.array(lin_vel)
        omega = np.array(ang_vel)
        vx, vy, vz = lin_vel
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        tilt_angle = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

        obs = self._get_obs()

        #                Objective 1: precision shaping
        r_precision = 0.0

        # 1. Uprightness: strong penalty for tilt, small bonus for very upright
        r_precision += -6.0 * tilt_angle
        if tilt_angle < 0.2:
            r_precision += 0.5 * (0.2 - tilt_angle)

        # 2. Horizontal drift: quadratic penalty for off-center
        r_precision += -0.2 * (x * x + y * y)

        # 3. Overall speed: gentle penalty
        r_precision += -0.15 * np.linalg.norm(v)

        # 4. Vertical speed: reward safe descent, penalize hovering/climbing/fast fall
        if vz < -0.1 and vz > -self.LAND_VZ_MAX:
            r_precision += 0.2  # small bonus for safe descent rate
        else:
            r_precision += -0.2 * abs(vz)  # penalize hovering, climbing, or fast fall

        # 5. Angular velocity damping
        r_precision += -0.3 * np.linalg.norm(omega)

        # 6. Constant time penalty to discourage hovering indefinitely
        r_precision += -0.15

        # 7. Near-ground control: force the descent to be arrested before the pad.
        if z < self.ALT_SWITCH:
            nearness = np.clip(
                (self.ALT_SWITCH - z) / (self.ALT_SWITCH - self.TOUCHDOWN_Z), 0.0, 1.0
            )
            vz_budget = self.LAND_VZ_MAX * (1.0 - 0.7 * nearness)

            r_precision += -(0.5 + 2.0 * nearness) * max(0.0, abs(vz) - vz_budget)

            r_precision += -0.3 * max(0.0, abs(vx) - self.LAND_VHORIZ_MAX)
            r_precision += -0.3 * max(0.0, abs(vy) - self.LAND_VHORIZ_MAX)

        r_precision_shaping = float(r_precision)

        #                    Terminal: failure vs success
        r_fail = 0.0  # applied to precision AND fuel
        r_success = 0.0  # applied to precision only
        terminated = False
        truncated = False
        term_reason = ""

        contacts = p.getContactPoints(self.rocketId, self.planeId)
        touched = len(contacts) > 0 or z <= self.TOUCHDOWN_Z

        if abs(x) > self.MAX_X or abs(y) > self.MAX_Y or z > self.MAX_Z:
            terminated = True
            term_reason = "OOB"
            r_fail += self.OOB_PENALTY
        elif tilt_angle > self.TILT_LIMIT:
            terminated = True
            term_reason = "TIP"
            r_fail += self.TIP_PENALTY
        elif touched:
            terminated = True
            soft = (
                abs(vz) < self.LAND_VZ_MAX
                and abs(vx) < self.LAND_VHORIZ_MAX
                and abs(vy) < self.LAND_VHORIZ_MAX
                and tilt_angle < self.LAND_TILT_MAX
                and np.sqrt(x * x + y * y) < self.LAND_RADIUS_MAX
            )
            if soft:
                term_reason = "LAND"
                vz_margin = 1.0 - min(abs(vz) / self.LAND_VZ_MAX, 1.0)
                vh_margin = 1.0 - min(max(abs(vx), abs(vy)) / self.LAND_VHORIZ_MAX, 1.0)
                tilt_margin = 1.0 - min(tilt_angle / self.LAND_TILT_MAX, 1.0)
                pos_margin = 1.0 - min(np.hypot(x, y) / self.LAND_RADIUS_MAX, 1.0)
                quality = 0.25 * (vz_margin + vh_margin + tilt_margin + pos_margin)
                r_success += self.LAND_BASE_BONUS + self.LAND_QUALITY_BONUS * quality

            else:
                r_fail += self.CRASH_PENALTY
                term_reason = "CRASH"

        if self.steps >= self.MAX_STEPS:
            truncated = True
            term_reason = "TIMEOUT"

        r_terminal = r_success + r_fail
        r_precision += r_terminal

        #                    Objective 2: fuel economy
        throttle_frac = T_mag / self.MAX_THRUST
        r_fuel = -self.FUEL_COST_SCALE * throttle_frac - self.FUEL_DEPLETION_CONSTANT
        r_fuel += r_fail

        self._ep_prec += float(r_precision)
        self._ep_fuel += float(r_fuel)
        self._ep_return += float(r_precision + r_fuel)

        if terminated or truncated:
            self._ret_window.append(self._ep_return)
            self._land_window.append(1.0 if term_reason == "LAND" else 0.0)
            ep_return_final = self._ep_return
            ep_prec_final = self._ep_prec
            ep_fuel_final = self._ep_fuel
            self._ep_return = 0.0
            self._ep_prec = 0.0
            self._ep_fuel = 0.0

            episode_stats = {
                "term_reason": term_reason,
                "steps": self.steps,
                "z": z,
                "vz": vz,
                "vx": vx,
                "vy": vy,
                "x": x,
                "y": y,
                "tilt_deg": np.degrees(tilt_angle),
                "pitch": np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))),
                "r_precision": ep_prec_final,
                "r_fuel": ep_fuel_final,
                "T_mag": T_mag,
                "pref_precision": self._ep_pref[0],
                "pref_fuel": self._ep_pref[1],
            }
            self._log_buffer.append(episode_stats)

            if self.episode_count % self.LOG_EVERY == 0 and self._log_buffer:
                n = len(self._log_buffer)
                avg = lambda key: sum(e[key] for e in self._log_buffer) / n
                outcomes = [e["term_reason"] for e in self._log_buffer]
                land_pct = outcomes.count("LAND") / n * 100
                crash_pct = outcomes.count("CRASH") / n * 100
                tip_pct = outcomes.count("TIP") / n * 100
                oob_pct = outcomes.count("OOB") / n * 100
                tout_pct = outcomes.count("TIMEOUT") / n * 100

                logging.info(
                    f"--- Ep {self.episode_count:>6} | Timestep ~{self.total_steps:>8,} | Last {n} episodes ---\n"
                    f"  Outcomes : LAND={land_pct:.1f}% CRASH={crash_pct:.1f}% "
                    f"TIP={tip_pct:.1f}% OOB={oob_pct:.1f}% TIMEOUT={tout_pct:.1f}%\n"
                    f"  Avg steps: {avg('steps'):.1f} | "
                    f"Avg r_precision: {avg('r_precision'):.2f} | "
                    f"Avg r_fuel: {avg('r_fuel'):.2f}\n"
                    f"  Avg z={avg('z'):.2f} vz={avg('vz'):.2f} "
                    f"tilt={avg('tilt_deg'):.1f}° pitch={avg('pitch'):.1f}°\n"
                    f"  Avg x={avg('x'):.2f} y={avg('y'):.2f} "
                    f"vx={avg('vx'):.2f} vy={avg('vy'):.2f} "
                    f"T_mag={avg('T_mag'):.1f}"
                )

                roll_ret = sum(self._ret_window) / max(1, len(self._ret_window))
                roll_land = (
                    sum(self._land_window) / max(1, len(self._land_window)) * 100
                )
                logging.info(
                    f"  [SANITY] Roll-100 mean EP RETURN={roll_ret:.1f} | "
                    f"Roll-100 LAND%={roll_land:.1f} | "
                    f"last ep return={ep_return_final:.1f} "
                    f"(prec={ep_prec_final:.1f} fuel={ep_fuel_final:.1f})"
                )

                window_full = len(self._land_window) >= self._land_window.maxlen
                land_frac = sum(self._land_window) / max(1, len(self._land_window))

                if window_full:
                    self._land_rate_history.append(land_frac)

                hist = self._land_rate_history
                stable = (
                    len(hist) >= hist.maxlen
                    and all(lr >= self.PROMOTE_LAND_PCT for lr in hist)
                    and (max(hist) - min(hist)) <= self.PROMOTE_STABILITY_BAND
                )

                if (
                    self.allow_auto_promote
                    and self.curriculum_level < self.MAX_CURRICULUM_LEVEL
                    and window_full
                    and self._episodes_at_level >= self.PROMOTE_MIN_EPISODES
                    and stable
                ):
                    old = self.curriculum_level
                    windows_snapshot = [round(r, 2) for r in hist]
                    self.curriculum_level = self.curriculum_level + 1
                    self._episodes_at_level = 0
                    self._land_window.clear()
                    self._ret_window.clear()
                    self._land_rate_history.clear()
                    logging.info(
                        f"  *** CURRICULUM PROMOTE {old} -> {self.curriculum_level} "
                        f"(windows={windows_snapshot} bar={self.PROMOTE_LAND_PCT}) ***"
                    )

                self._log_buffer.clear()

        info_dict = {
            "r_precision": float(r_precision),
            "r_fuel": float(r_fuel),
            "r_terminal": float(r_terminal),
            "shaping_reward": float(r_precision_shaping),
            "terminal_reward": float(r_terminal),
            "z": float(z),
            "vz": float(vz),
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "tilt_deg": float(np.degrees(tilt_angle)),
            "T_base": float(T_mag),
            "termination": term_reason,
            "episode_return": float(r_precision + r_fuel),
            "episode_length": self.steps,
            "pitch": float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))),
        }
        reward_vec = np.array([r_precision, r_fuel], dtype=np.float32)
        return obs, reward_vec, terminated, truncated, info_dict

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.rocketId)
        lin_vel, ang_vel = p.getBaseVelocity(self.rocketId)
        x, y, z = pos

        v_targ, t_go = self._compute_vtarg(pos, lin_vel)
        v_err = np.array(lin_vel) - v_targ

        qx, qy, qz, qw = orn
        if qw < 0.0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        wx, wy, wz = ang_vel

        return np.array(
            [
                x,
                y,
                v_err[0],
                v_err[1],
                v_err[2],
                qx,
                qy,
                qz,
                qw,
                wx,
                wy,
                wz,
                z,
                float(np.clip(t_go, 0.0, 50.0)),
            ],
            dtype=np.float32,
        )

    def _compute_vtarg(self, pos, vel):
        x, y, z = pos

        if z > self.ALT_SWITCH:
            r_hat = np.array([x, y, z - self.ALT_SWITCH])
            r_norm = np.linalg.norm(r_hat) + 1e-6
            v_offset = np.array([0.0, 0.0, -2.0])
            v_norm = np.linalg.norm(np.array(vel) - v_offset) + 1e-6
            t_go = r_norm / v_norm
            v_targ = (
                -self.V0 * (r_hat / r_norm) * (1.0 - np.exp(-t_go / self.TAU_HIGH))
                + v_offset
            )
        else:
            z_frac = np.clip(z / self.ALT_SWITCH, 0.0, 1.0)
            vz_targ = -self.LAND_VZ_MAX * z_frac
            vx_targ = -self.K_HORIZ * x
            vy_targ = -self.K_HORIZ * y
            v_targ = np.array([vx_targ, vy_targ, vz_targ])
            t_go = z / (abs(vz_targ) + 1e-6)

        return v_targ, t_go

    def close(self):
        p.disconnect()
