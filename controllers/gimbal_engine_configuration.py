import os
import time
import math
import random
import gymnasium as gym
import logging
import numpy as np
import pybullet as p
import pybullet_data
import csv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rocket_model.urdf"
LOGGER_PATH = BASE_DIR / "logs" / "pd_controller"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(str(LOGGER_PATH / "pd_gimbal_logs.log"), mode="w"),
        logging.StreamHandler(),
    ],
)

USE_GUI = False

physicsClient = p.connect(p.GUI if USE_GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

gravity = -10.0

p.setGravity(0, 0, gravity)

planeId = p.loadURDF("plane.urdf")
rocketId = p.loadURDF(
    str(MODEL_PATH),
    useFixedBase=False,
)


def get_rotation_matrix(quaternion):
    return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)


def sample_initial_conditions(cl):
    if cl == 0:
        pos = [
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(1.7, 2.5),
        ]
        eul = [0.0, random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]
        vel = [
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            random.uniform(-1.0, -0.3),
        ]
    elif cl == 1:
        pos = [
            random.uniform(-1.2, 1.2),
            random.uniform(-1.2, 1.2),
            random.uniform(2.5, 4.0),
        ]
        eul = [0.0, random.uniform(-0.07, 0.07), random.uniform(-0.07, 0.07)]
        vel = [
            random.uniform(-0.8, 0.8),
            random.uniform(-0.8, 0.8),
            random.uniform(-1.5, -0.4),
        ]
    elif cl == 2:
        pos = [
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(4.0, 6.0),
        ]
        eul = [0.0, random.uniform(-0.12, 0.12), random.uniform(-0.12, 0.12)]
        vel = [
            random.uniform(-1.2, 1.2),
            random.uniform(-1.2, 1.2),
            random.uniform(-2.5, -0.8),
        ]
    else:
        pos = [
            random.uniform(-3.0, 3.0),
            random.uniform(-3.0, 3.0),
            random.uniform(8.0, 12.0),
        ]
        eul = [0.0, random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)]
        vel = [random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-7.0, -3.0)]
    return pos, p.getQuaternionFromEuler(eul), vel


if USE_GUI:
    p.resetDebugVisualizerCamera(
        cameraDistance=6,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[10, -10, 15],
    )

dt = 1 / 240
log_interval = 24
frame_count = 0


dyn = p.getDynamicsInfo(rocketId, -1)
mass = dyn[0]
inertia = dyn[2]
Ixx, Iyy, Izz = inertia
r_t = 0.3
ell = 1.5
g = abs(gravity)

T_min, T_max = 0.0, 300.0

# Controller gains
omega_attitude = 4.0
omega_altitude = 1.5
omega_lateral = 1.0
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

z_target = 1.55
x_target = 0.0
y_target = 0.0

GIMBAL_MAX = np.radians(12.0)


def run_episode(curriculum_level, gui=False):
    frame_count = 0

    pos, orn, vel = sample_initial_conditions(curriculum_level)
    p.resetBasePositionAndOrientation(rocketId, pos, orn)
    p.resetBaseVelocity(rocketId, vel, [0, 0, 0])

    outcome = None

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

        T = float(np.clip(thrust_z, T_min, T_max))
        T_safe = max(T, 1e-3)

        delta_y = np.clip(tx / (T_safe * ell), -GIMBAL_MAX, GIMBAL_MAX)
        delta_x = np.clip(-ty / (T_safe * ell), -GIMBAL_MAX, GIMBAL_MAX)

        Fx = T * math.sin(delta_x)
        Fy = T * math.sin(delta_y)
        Fz = T * math.cos(delta_x) * math.cos(delta_y)

        p.applyExternalForce(
            objectUniqueId=rocketId,
            linkIndex=-1,
            forceObj=[Fx, Fy, Fz],
            posObj=[0, 0, -1.5],
            flags=p.LINK_FRAME,
        )

        # Landing Detection
        tilt_angle = float(np.arccos(np.clip(R[2][2], -1.0, 1.0)))
        horiz_r = math.sqrt(x**2 + y**2)
        contacts = p.getContactPoints(rocketId, planeId)
        touched = len(contacts) > 0 or z <= 1.55

        outcome = None
        if abs(x) > 10.0 or abs(y) > 10.0 or z > 15.0:
            outcome = "OOB"
        elif tilt_angle > 0.5:
            outcome = "TIP"
        elif touched:
            soft = (
                abs(vz) < 1.5
                and abs(vx) < 1.0
                and abs(vy) < 1.0
                and tilt_angle < 0.30
                and horiz_r < 1.0
            )
            outcome = "LAND" if soft else "CRASH"
        elif frame_count >= 4800:
            outcome = "TIMEOUT"

        if outcome is not None:
            logging.info(
                f"Outcome: {outcome} || Pos: {newPos[0]:.4f}, {newPos[1]:.4f}, {newPos[2]:.4f} | vz: {lin_vel[2]:.4f} | P/R: {eul[0]:.4f}, {eul[1]:.4f}"
            )
            p.resetBaseVelocity(rocketId, [0, 0, 0], [0, 0, 0])
            break

        p.stepSimulation()
        if USE_GUI:
            time.sleep(dt)
    return outcome


results = {
    cl: {"LAND": 0, "CRASH": 0, "TIP": 0, "OOB": 0, "TIMEOUT": 0} for cl in range(4)
}
N_TRIALS = 500
for cl in range(4):
    logging.info(f"Current curriculum level: {cl}")
    for _ in range(N_TRIALS):
        outcome = run_episode(curriculum_level=cl)
        results[cl][outcome] += 1


p.disconnect()

with open(
    LOGGER_PATH / f"pd_outputs_trials_{N_TRIALS}.csv", "w", newline=""
) as csvfile:
    csv_writer = csv.DictWriter(csvfile, results.keys())
    csv_writer.writeheader()
    csv_writer.writerow(results)
