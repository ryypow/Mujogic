"""
Test script to run one episode and observe the behavior with random actions.
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
# State translator (simplified - no dist_bin)
# =============================
def state_translator(observation, env):

    base_env = env.unwrapped #Gym stores the unwrapped environment in .unwrapped -> needed for palm position


    # ==== discrete bin for progress towards goal
    cube_quart = observation[-4:] #quartnerion orientatoin for cube
    r = Rotation.from_quat(cube_quart) #reminder: working with radians
    euler = r.as_euler('xyz') #x,y,z coordinates
    z_rotation = euler[2]    

    goal = base_env.target_rotation
    goal_progress = abs(goal - z_rotation)

    if goal_progress > np.deg2rad(30):
        progress_bin = 0 #far from goal
    elif goal_progress > np.deg2rad(15):
        progress_bin = 1 #getting there
    elif goal_progress > np.deg2rad(5): #agreeing with configured tolerance
        progress_bin = 2 #acceptable
    else:
        progress_bin = 3 #winner winner chicken dinner

    #testing grasp strength - fingers in contact (from reward function)
    fingers_in_contact = set()  # the reward function uses a set
    
    for i in range(base_env.sim.data.ncon):
        contact = base_env.sim.data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        
        if geom1 in base_env.fingertip_geom_ids and geom2 == base_env.can_geom_id:
            fingers_in_contact.add(geom1)
        elif geom2 in base_env.fingertip_geom_ids and geom1 == base_env.can_geom_id:
            fingers_in_contact.add(geom2)
    
    num_fingers = len(fingers_in_contact)
    
    if num_fingers <= 1:
        grasp_bin = 0  # weak
    elif num_fingers == 2:
        grasp_bin = 1  # stable
    else:  # >= 3
        grasp_bin = 2  # strong


    #==== discrete bin for speed
    #rotation velocity - from env.calculate_reward()
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(base_env.sim.model, base_env.sim.data, mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0)
    angular_velocity_z = abs(obj_vel[2]) #spin speed

    #rotation speed bins - unsure about actual speed, adjust accordingly TODO.txt
    if angular_velocity_z < 0.1:
        speed_bin = 0 #not spinning
    elif angular_velocity_z < 0.5:
        speed_bin = 1 #rotating
    else:
        speed_bin = 2 #rotating fast
        
    state_id = grasp_bin * 12 + speed_bin * 4 + progress_bin

    return state_id, progress_bin, grasp_bin, speed_bin, z_rotation, goal_progress

# =============================
# Parameters
# =============================
GOAL = 90  # Target rotation in degrees
MAX_STEPS = 300
NUM_ACTIONS = 4
ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROTATE",
    3: "HOLD"
}

# =============================
# Initialize environment
# =============================
print("=" * 60)
print("ONE EPISODE TEST - 4-Action Discrete Translator")
print("=" * 60)
print(f"\nTarget rotation: {GOAL} degrees ({np.deg2rad(GOAL):.3f} radians)")
print(f"Max steps: {MAX_STEPS}")
print(f"Actions: {list(ACTION_NAMES.values())}")
print(f"States: 12 (3 speed × 4 progress)")  # Update this if you changed state_translator
print("\nInitializing environment with human rendering...")

base_env = CanRotateEnv(GOAL, render_mode="headless")
env = ActionTranslator(base_env)

print("Environment initialized!")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# =============================
# Run one episode
# =============================
print("\n" + "=" * 60)
print("STARTING EPISODE")
print("=" * 60)

observation, _ = env.reset()
state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

print(f"\nInitial state:")
print(f"  State ID: {state}")
print(f"  Z-rotation: {np.rad2deg(z_rot):.2f} deg")
print(f"  Goal progress: {np.rad2deg(goal_prog):.2f} deg away")
print(f"  Speed bin: {speed_bin} (0=still, 1=slow, 2=fast)")
print(f"  Progress bin: {progress_bin} (0=far, 1=mid, 2=close, 3=done)")

total_reward = 0.0
step = 0

print("\n" + "-" * 70)
print("Step | Action   | State | Z-Rot  | Speed | Progress | Reward")
print("-" * 70)

try:
    while step < MAX_STEPS:
        # Random action for testing
        action = 2

        # Execute action
        next_observation, reward, terminated, truncated, _ = env.step(action)

        # Get new state info
        new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

        total_reward += reward
        step += 1

        # Extract z_rotation to verify direction
        cube_quat = next_observation[-4:]  # [qw, qx, qy, qz]
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]])
        z_rotation = r.as_euler('xyz')[2]

        print(f"Step {step}: z_rotation = {np.rad2deg(z_rotation):.2f}°")


        # Print every 10 steps to avoid flooding
        if step % 10 == 0 or terminated or truncated:
            action_name = ACTION_NAMES.get(action, "???")
            print(f"{step:4d} | {action_name:8s} | {new_state:5d} | {np.rad2deg(z_rot):6.1f} | {speed_bin:5d} | {progress_bin:8d} | {reward:7.2f}")

        state = new_state

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
