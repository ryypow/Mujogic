from inhand_env import CanRotateEnv
from discrete_translator import ActionTranslator
import numpy as np

env = ActionTranslator(CanRotateEnv(target_degrees=45, render_mode="human"))
obs, _ = env.reset()

print("Testing GRASP macro...")
for step in range(60):
    obs, r, term, trunc, info = env.step(0)  # GRASP
    if term or trunc:
        break

print("Testing ROT_POS macro...")
for step in range(120):
    obs, r, term, trunc, info = env.step(2)  # ROT_POS
    if term or trunc:
        break

print("Testing ROT_NEG macro...")
for step in range(120):
    obs, r, term, trunc, info = env.step(4)  # ROT_NEG
    if term or trunc:
        break

import mujoco
base_env = env.unwrapped
print("Hand joints and pose indices:")
for idx, j_id in enumerate(base_env.sim.hand_joint_ids):
    name = mujoco.mj_id2name(
        base_env.sim.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        j_id,
    )
    print(idx, j_id, name)
