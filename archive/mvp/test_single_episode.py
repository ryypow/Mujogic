"""
Test script to run one episode and observe the Q-agent behavior.
This will show the simulation visually so you can see what's happening.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from discrete_translator import ActionTranslator
from inhand_env import CanRotateEnv
import mujoco

# =============================
# State translator (from q_agent.py)
# =============================
def state_translator(observation, env):
    """Convert continuous observation to discrete state ID"""
    base_env = env.unwrapped

    # Discrete bin for progress towards goal
    cube_quart = observation[-4:]
    r = Rotation.from_quat(cube_quart)
    euler = r.as_euler('xyz')
    z_rotation = euler[2]

    goal = base_env.target_rotation
    goal_progress = abs(goal - z_rotation)

    if goal_progress > np.deg2rad(60):
        progress_bin = 0
    elif goal_progress > np.deg2rad(30):
        progress_bin = 1
    elif goal_progress > np.deg2rad(10):
        progress_bin = 2
    else:
        progress_bin = 3

    # Discrete bin for distance from cube to palm
    palm_position = base_env.sim.data.site_xpos[base_env.site_id]
    cube_position = observation[16:19]
    cube_distance_fromPalm = np.linalg.norm(cube_position - palm_position)

    if cube_distance_fromPalm < 0.05:
        dist_bin = 0
    elif cube_distance_fromPalm < 0.10:
        dist_bin = 1
    else:
        dist_bin = 2

    # Discrete bin for speed
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(base_env.sim.model, base_env.sim.data, mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0)
    angular_velocity_z = obj_vel[2]

    if angular_velocity_z < 0.5:
        speed_bin = 0
    elif angular_velocity_z < 2.0:
        speed_bin = 1
    else:
        speed_bin = 2

    state_id = dist_bin * 12 + speed_bin * 4 + progress_bin
    return state_id, progress_bin, dist_bin, speed_bin, z_rotation, goal_progress

# =============================
# Parameters
# =============================
GOAL = 90  # Target rotation in degrees
MAX_STEPS = 300
ACTION_NAMES = {0: "GRASP", 1: "ROTATE", 2: "HOLD"}

# =============================
# Initialize environment
# =============================
print("=" * 60)
print("ONE EPISODE TEST - Q-Agent with Discrete Translator")
print("=" * 60)
print(f"\nTarget rotation: {GOAL} degrees ({np.deg2rad(GOAL):.3f} radians)")
print(f"Max steps: {MAX_STEPS}")
print("\nInitializing environment with human rendering...")

base_env = CanRotateEnv(GOAL, render_mode="human")
env = ActionTranslator(base_env)

print("Environment initialized!")
print(f"Action space: {env.action_space} (0=GRASP, 1=ROTATE, 2=HOLD)")
print(f"Observation space: {env.observation_space}")

# =============================
# Run one episode
# =============================
print("\n" + "=" * 60)
print("STARTING EPISODE")
print("=" * 60)

observation, _ = env.reset()
state, progress_bin, dist_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

print(f"\nInitial state:")
print(f"  State ID: {state}")
print(f"  Z-rotation: {np.rad2deg(z_rot):.2f} deg")
print(f"  Goal progress: {np.rad2deg(goal_prog):.2f} deg away")
print(f"  Distance bin: {dist_bin} (0=close, 1=med, 2=far)")
print(f"  Speed bin: {speed_bin} (0=still, 1=slow, 2=fast)")
print(f"  Progress bin: {progress_bin} (0=far, 1=mid, 2=close, 3=done)")

total_reward = 0.0
step = 0

print("\n" + "-" * 60)
print("Step | Action | State | Z-Rot | Dist | Speed | Progress | Reward")
print("-" * 60)

try:
    while step < MAX_STEPS:
        # Random action for testing (you can change this)
        action = env.action_space.sample()

        # Execute action
        next_observation, reward, terminated, truncated, _ = env.step(action)

        # Get new state info
        new_state, progress_bin, dist_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

        total_reward += reward
        step += 1

        # Print every 10 steps to avoid flooding
        if step % 10 == 0 or terminated or truncated:
            print(f"{step:4d} | {ACTION_NAMES[action]:6s} | {new_state:5d} | {np.rad2deg(z_rot):6.1f} | {dist_bin:4d} | {speed_bin:5d} | {progress_bin:8d} | {reward:7.2f}")

        if terminated:
            print(f"\n*** TERMINATED at step {step} ***")
            if goal_prog < np.deg2rad(5):
                print("SUCCESS! Target rotation achieved!")
            else:
                print("Cube dropped or other termination condition")
            break

        if truncated:
            print(f"\n*** TRUNCATED at step {step} (max steps reached) ***")
            break

        state = new_state

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

print("\n" + "=" * 60)
print("EPISODE SUMMARY")
print("=" * 60)
print(f"Total steps: {step}")
print(f"Total reward: {total_reward:.2f}")
print(f"Final Z-rotation: {np.rad2deg(z_rot):.2f} degrees")
print(f"Goal was: {GOAL} degrees")
print(f"Final distance from goal: {np.rad2deg(goal_prog):.2f} degrees")

# Clean up
print("\nClosing environment...")
env.close()
print("Done!")
