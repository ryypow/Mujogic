"""
Diagnostic: Check cube rotation at each stage of reset()
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from inhand_env import CanRotateEnv
import mujoco

def get_cube_z_rotation(env):
    """Get current z-rotation of cube in degrees"""
    obj_jnt_adr = env.sim.model.body_jntadr[env.obj_body_id]
    obj_qpos_adr = env.sim.model.jnt_qposadr[obj_jnt_adr]
    quat = env.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7]
    r = Rotation.from_quat(quat)
    euler = r.as_euler('xyz')
    return np.rad2deg(euler[2])

print("=" * 60)
print("DIAGNOSTIC: Tracking cube rotation during reset()")
print("=" * 60)

# Create env without calling reset yet
env = CanRotateEnv(90, render_mode=None)

# Now manually step through reset process
print("\n1. After mj_resetData:")
mujoco.mj_resetData(env.sim.model, env.sim.data)
mujoco.mj_forward(env.sim.model, env.sim.data)
print(f"   Z-rotation: {get_cube_z_rotation(env):.2f} deg")

# Set arm position
target_pos_up = np.array([0.4, 0.0, .5])
target_euler_up = np.array([0, 0, 0])
q_palm_up = env.sim.desired_qpos_from_ik(env.site_id, target_pos_up, target_euler_up)
env.sim.set_joint_positions(env.sim.arm_joint_ids, q_palm_up)
for i, act_id in enumerate(env.sim.arm_act_ids):
    env.sim.data.ctrl[act_id] = q_palm_up[i]
mujoco.mj_forward(env.sim.model, env.sim.data)

print("\n2. After setting arm position:")
print(f"   Z-rotation: {get_cube_z_rotation(env):.2f} deg")

# Set cube position and quaternion
palm_surface_pos = env.sim.data.site_xpos[env.site_id].copy()
object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075])
obj_jnt_adr = env.sim.model.body_jntadr[env.obj_body_id]
obj_qpos_adr = env.sim.model.jnt_qposadr[obj_jnt_adr]
env.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos
env.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0]  # Identity quaternion
mujoco.mj_forward(env.sim.model, env.sim.data)

print("\n3. After setting cube position (quat = [1,0,0,0]):")
print(f"   Z-rotation: {get_cube_z_rotation(env):.2f} deg")

# The 20 settling steps
print("\n4. During 20 settling steps:")
for i in range(20):
    mujoco.mj_step(env.sim.model, env.sim.data)
    if i % 5 == 0 or i == 19:
        print(f"   Step {i+1}: Z-rotation = {get_cube_z_rotation(env):.2f} deg")

# Set hand angles
q_open_angles = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.3, 1.0, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0
])
env.sim.set_joint_positions(env.sim.hand_joint_ids, q_open_angles)
for i, act_id in enumerate(env.sim.hand_act_ids):
    env.sim.data.ctrl[act_id] = q_open_angles[i]
mujoco.mj_forward(env.sim.model, env.sim.data)

print("\n5. After setting hand 'open' angles:")
print(f"   Z-rotation: {get_cube_z_rotation(env):.2f} deg")

# Now call actual reset and compare
print("\n" + "=" * 60)
print("6. After full env.reset():")
obs, _ = env.reset()
cube_quat = obs[-4:]
r = Rotation.from_quat(cube_quat)
euler = r.as_euler('xyz')
print(f"   Z-rotation: {np.rad2deg(euler[2]):.2f} deg")
print("=" * 60)

env.close()
print("\nDone!")
