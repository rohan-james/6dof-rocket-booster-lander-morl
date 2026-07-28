import os
import time
import logging
import random
import sys
import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from pathlib import Path
from gymnasium import spaces

BASE_DIR = Path(__file__).resolve().parent.parent
LOGGER_PATH = BASE_DIR / "logs" / "sorl"
MODEL_PATH = BASE_DIR / "models" / "rocket_model.urdf"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(LOGGER_PATH) / "6dof_logs.log", mode="w"),
        logging.StreamHandler(),
    ],
)


class RocketLanding6DOFEnv(gym.Env):
    GRAVITY = 9.80
    MIN_THRUST, MAX_THRUST = (
        0.0,
        300.0,
    )
    MAX_GIMBAL_DEG = 12.0

    SIM_DT = 1.0 / 240.0
    N_SUBSTEPS = 12
    MAX_STEPS = 400

    BASE_HALF_LEN = 1.5
    TOUCHDOWN_Z = BASE_HALF_LEN + 0.05

    # landing gates
    LAND_VZ_MAX = 1.5
    LAND_VHORIZ_MAX = 1.0
    LAND_TILT_MAX = 0.30  # rad ≈ 17°

    # safety / bounds
    TILT_LIMIT = 0.50
    TILT_WARN = 0.60
    MAX_X = 10.0
    MAX_Y = 10.0
    MAX_Z = 15.0

    # velocity-field guidance
    TAU = 1.2  # s, target-velocity time constant
    V0 = 5.0  # m/s, characteristic approach speed

    # reward terminals
    LANDING_BONUS = 100.0
    CRASH_PENALTY = -100.0
    OOB_PENALTY = -100.0
    TIP_PENALTY = -100.0

    ALT_SWITCH = 3.0
    TAU_HIGH = 1.2
    TAU_LOW = 0.6
    LAND_RADIUS_MAX = 1.0

    V_LOW = 2.0
    K_HORIZ = 0.5

    def __init__(self, render_mode=None, curriculum_level=0):
        super().__init__()
        self.urdf_path = str(MODEL_PATH)
        self.render_mode = render_mode
        self.curriculum_level = curriculum_level

        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=50,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 3],
            )
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.GRAVITY)
        self.planeId = p.loadURDF("plane.urdf")
        self.rocketId = p.loadURDF(self.urdf_path, useFixedBase=False)

        for i in range(p.getNumJoints(self.rocketId)):
            if p.getJointInfo(self.rocketId, i)[1].decode() == "thrust_joint":
                self.thrust_link_index = i
                break
        self.mass = p.getDynamicsInfo(self.rocketId, -1)[0]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.steps = 0

        self.reset()

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

        cl = self.curriculum_level
        if cl == 0:
            startPos = [
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(1.7, 2.5),
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
            ]  # was 2.5–5.0
            startEuler = [
                0.0,
                random.uniform(-0.07, 0.07),
                random.uniform(-0.07, 0.07),
            ]
            startVel = [
                random.uniform(-0.8, 0.8),
                random.uniform(-0.8, 0.8),
                random.uniform(-1.5, -0.4),
            ]  # was -2.0 to -0.5
        elif cl == 2:
            startPos = [
                random.uniform(-2.0, 2.0),
                random.uniform(-2.0, 2.0),
                random.uniform(4.0, 6.0),
            ]
            startEuler = [0.0, random.uniform(-0.12, 0.12), random.uniform(-0.12, 0.12)]
            startVel = [
                random.uniform(-1.2, 1.2),
                random.uniform(-1.2, 1.2),
                random.uniform(-2.5, -0.8),
            ]
        else:
            startPos = [
                random.uniform(-2.5, 2.5),
                random.uniform(-2.5, 2.5),
                random.uniform(6.0, 8.5),
            ]
            startEuler = [0.0, random.uniform(-0.16, 0.16), random.uniform(-0.16, 0.16)]
            startVel = [
                random.uniform(-1.8, 1.8),
                random.uniform(-1.8, 1.8),
                random.uniform(-3.5, -1.2),
            ]

        startOrn = p.getQuaternionFromEuler(startEuler)
        p.resetBasePositionAndOrientation(self.rocketId, startPos, startOrn)
        p.resetBaseVelocity(self.rocketId, startVel, [0, 0, 0])
        self._null_roll_rate()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

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

        # shaping reward
        r_shaping = (
            -0.1 * np.sqrt(x * x + y * y)  # horizontal position
            - 0.1 * abs(z - 0.2)  # altitude tracking
            - 0.2 * np.linalg.norm(v)  # overall speed
            - 5.0 * tilt_angle  # upright incentive
            - 0.3 * np.linalg.norm(omega)  # rotation penalty
            + (0.1 if vz < -0.1 else 0.0)  # progress
        )

        if tilt_angle > self.TILT_WARN:
            r_shaping += -6.0 * (tilt_angle - self.TILT_WARN)

        r_shaping += 0.5 * (np.cos(tilt_angle))

        # near-ground: demand slow, on-pad terminal approach
        if z < self.ALT_SWITCH:
            r_shaping += -0.5 * max(0.0, abs(vz) - self.LAND_VZ_MAX)
            r_shaping += -0.5 * max(0.0, abs(vx) - self.LAND_VHORIZ_MAX)
            r_shaping += -0.5 * max(0.0, abs(vy) - self.LAND_VHORIZ_MAX)
            r_shaping += -0.4 * np.sqrt(x * x + y * y)
            r_shaping += 1.0 * max(0.0, 1.0 - abs(vz) / self.LAND_VZ_MAX)

            if z > self.TOUCHDOWN_Z + 0.1 and abs(vz) < 0.3:
                r_shaping += -0.5

        r_terminal = 0.0
        terminated = False
        truncated = False
        term_reason = ""

        contacts = p.getContactPoints(self.rocketId, self.planeId)
        touched = len(contacts) > 0 or z <= self.TOUCHDOWN_Z

        if abs(x) > self.MAX_X or abs(y) > self.MAX_Y or z > self.MAX_Z:
            terminated = True
            term_reason = "OOB"
            r_terminal += self.OOB_PENALTY

        elif tilt_angle > self.TILT_LIMIT:
            terminated = True
            term_reason = "TIP"
            r_terminal += self.TIP_PENALTY

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
                r_terminal += self.LANDING_BONUS
                term_reason = "LAND"
            else:
                r_terminal += self.CRASH_PENALTY
                term_reason = "CRASH"

        if self.steps >= 120:
            truncated = True
            term_reason = "TIMEOUT"
            r_terminal -= 100

        reward = r_shaping + r_terminal

        info_dict = {
            "shaping_reward": float(r_shaping),
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
            "episode_return": float(reward),
            "episode_length": self.steps,
            "pitch": float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))),
        }
        return obs, reward, terminated, truncated, info_dict

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
            vz_targ = -self.V_LOW * z_frac
            vx_targ = -self.K_HORIZ * x
            vy_targ = -self.K_HORIZ * y
            v_targ = np.array([vx_targ, vy_targ, vz_targ])
            t_go = z / (abs(vz_targ) + 1e-6)

        return v_targ, t_go

    def close(self):
        p.disconnect()
