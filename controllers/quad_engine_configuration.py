import os
import time
import math
import random
import gymnasium as gym
import logging
import numpy as np
import pybullet as p
import pybullet_data

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rocket_model.urdf"
LOGGER_PATH = BASE_DIR / "logs" / "pd_controller"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(LOGGER_PATH / "pd_quad_logs.log"), mode="w"),
        logging.StreamHandler(),
    ],
)

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

gravity = -10.0  # Earth


p.setGravity(0, 0, gravity)

planeId = p.loadURDF("plane.urdf")
rocketId = p.loadURDF(
    str(MODEL_PATH),
    useFixedBase=False,
)


def get_rotation_matrix(quaternion):
    return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)


startPos = [
    random.uniform(-2.5, 2.5),
    random.uniform(-2.5, 2.5),
    random.uniform(5, 8),
]
startOrientation = p.getQuaternionFromEuler(
    [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.0]
)

p.resetBasePositionAndOrientation(rocketId, startPos, startOrientation)
p.resetDebugVisualizerCamera(
    cameraDistance=6, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 2]
)

dt = 1 / 240
log_interval = 24
frame_count = 0


dyn = p.getDynamicsInfo(rocketId, -1)
mass = dyn[0]
inertia = dyn[2]
Ixx, Iyy, Izz = inertia
r_t = 0.3
g = abs(gravity)

T_min, T_max = 0, 150

# Controller gains
omega_attitude = 4.0
omega_altitude = 0.5
omega_lateral = 0.5
zeta = 1.2

# Attitude controller
Kp_roll = Ixx * omega_attitude**2
Kp_pitch = Iyy * omega_attitude**2

Kd_roll = 2 * Ixx * zeta * omega_attitude
Kd_pitch = 2 * Iyy * zeta * omega_attitude

# Altitude controller
Kp_z = mass * omega_altitude**2
Kd_z = 2 * mass * zeta * omega_altitude

# Lateral controller
Kp_x = omega_lateral**2
Kd_x = 2 * zeta * omega_lateral

Kp_y = omega_lateral**2
Kd_y = 2 * zeta * omega_lateral

z_target = 0.2
x_target = 0.0
y_target = 0.0

LANDING_LOCK = False

while True:
    frame_count += 1

    newPos, newOrn = p.getBasePositionAndOrientation(rocketId)
    lin_vel, ang_vel = p.getBaseVelocity(rocketId)
    eul = p.getEulerFromQuaternion(newOrn)
    roll, pitch, yaw = p.getEulerFromQuaternion(newOrn)

    x, y, z = newPos
    vx, vy, vz = lin_vel
    wx, wy, wz = ang_vel

    rot_matrix = p.getMatrixFromQuaternion(newOrn)

    R = [
        [rot_matrix[0], rot_matrix[1], rot_matrix[2]],
        [rot_matrix[3], rot_matrix[4], rot_matrix[5]],
        [rot_matrix[6], rot_matrix[7], rot_matrix[8]],
    ]

    p_rate = R[0][0] * wx + R[1][0] * wy + R[2][0] * wz
    q_rate = R[0][1] * wx + R[1][1] * wy + R[2][1] * wz
    r_rate = R[0][2] * wx + R[1][2] * wy + R[2][2] * wz

    # Vertical difference
    z_err = z_target - z
    vz_err = -vz

    # Lateral difference
    x_err = x_target - x
    vx_err = -vx
    y_err = y_target - y
    vy_err = -vy

    R22 = max(0.3, R[2][2])
    thrust_z = (mass * g + Kp_z * z_err + Kd_z * vz_err) / R22

    ax_des = Kp_x * x_err + Kd_x * vx_err  # desired x-acceleration
    ay_des = Kp_y * y_err + Kd_y * vy_err  # desired y-acceleration
    TILT_MAX = 0.3
    theta_desired = np.clip(
        math.atan2(ax_des, g), -TILT_MAX, TILT_MAX
    )  # tilt toward +x => negative pitch
    phi_desired = np.clip(
        -math.atan2(ay_des, g), -TILT_MAX, TILT_MAX
    )  # tilt toward +y => positive roll

    # Attitude control
    roll_err = phi_desired - roll
    pitch_err = theta_desired - pitch

    tx = Kp_roll * roll_err - Kd_roll * p_rate
    ty = Kp_pitch * pitch_err - Kd_pitch * q_rate

    if not LANDING_LOCK:
        T_base = thrust_z / 4

        T1 = np.clip(T_base + (tx / (2 * r_t)), T_min, T_max)
        T2 = np.clip(T_base - (tx / (2 * r_t)), T_min, T_max)
        T3 = np.clip(T_base - (ty / (2 * r_t)), T_min, T_max)
        T4 = np.clip(T_base + (ty / (2 * r_t)), T_min, T_max)

        thrust_position_map = [
            (T1, [0, 0.3, -1.5]),
            (T2, [0, -0.3, -1.5]),
            (T3, [0.3, 0, -1.5]),
            (T4, [-0.3, 0, -1.5]),
        ]
        if frame_count % log_interval == 0:
            logging.info(
                f"Thruster 1: {T1:.3f} | Thruster 2: {T2:.3f} | Thruster 3: {T3:.3f} | Thruster 4: {T4:.3f}"
            )

        for engine_thrust, throttle_position in thrust_position_map:
            p.applyExternalForce(
                objectUniqueId=rocketId,
                linkIndex=-1,
                forceObj=[0, 0, engine_thrust],
                posObj=throttle_position,
                flags=p.LINK_FRAME,
            )

    # Landing Detection
    landed = z < 1.51 and abs(vz) < 0.15 and abs(roll) < 0.05 and abs(pitch) < 0.05

    if landed and not LANDING_LOCK:
        LANDING_LOCK = True
        p.resetBaseVelocity(rocketId, [0, 0, 0], [0, 0, 0])
        logging.info("Landing Complete")
        break

    if frame_count % log_interval == 0:
        logging.info(
            f"Pos: {newPos[0]:.4f}, {newPos[1]:.4f}, {newPos[2]:.4f} | vz: {lin_vel[2]:.4f} | P/R: {eul[0]:.4f}, {eul[1]:.4f}"
        )

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()
