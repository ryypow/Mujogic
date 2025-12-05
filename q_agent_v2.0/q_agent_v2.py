"""
Q-Learning V2.1 for cube rotation with FINER progress bins
- 54 states (3 grasp x 3 speed x 6 progress)
- 6 progress bins for finer granularity (was 4)
- Relative progress bins that scale to any goal (45, 60, 90, etc.)
- Curriculum learning: start small, increase goal progressively
"""
import sys
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from discrete_translator import ActionTranslator
from inhand_env import CanRotateEnv


def state_translator(observation, env):
    """
    Convert continuous observation to discrete state ID.
    Uses RELATIVE progress bins that scale to any goal size.
    V2.1: Increased to 6 progress bins for finer control.
    """
    base_env = env.unwrapped

    # Get current Z rotation from quaternion
    cube_quat_mujoco = observation[-4:]  # MuJoCo format: [w, x, y, z]
    cube_quat_scipy = np.array([
        cube_quat_mujoco[1], cube_quat_mujoco[2],
        cube_quat_mujoco[3], cube_quat_mujoco[0]
    ])  # SciPy format: [x, y, z, w]
    r = Rotation.from_quat(cube_quat_scipy)
    z_rotation = r.as_euler('xyz')[2]

    # Goal progress (how far from target)
    goal = base_env.target_rotation
    goal_progress = abs(goal - z_rotation)
    goal_abs = abs(base_env.rotation_goal_delta)  # Use delta, not absolute target

    # RELATIVE progress bins - 6 bins for finer control
    if goal_abs > 0:
        progress_ratio = goal_progress / goal_abs
    else:
        progress_ratio = 0

    # 6 progress bins (was 4) - more granular for better learning
    if progress_ratio > 0.80:
        progress_bin = 0  # very far (>80% remaining)
    elif progress_ratio > 0.60:
        progress_bin = 1  # far (60-80% remaining)
    elif progress_ratio > 0.40:
        progress_bin = 2  # medium (40-60% remaining)
    elif progress_ratio > 0.20:
        progress_bin = 3  # close (20-40% remaining)
    elif progress_ratio > 0.10:
        progress_bin = 4  # very close (10-20% remaining)
    else:
        progress_bin = 5  # at goal (<10% remaining)

    # Grasp strength bin (finger contact)
    fingers_in_contact = set()
    for i in range(base_env.sim.data.ncon):
        contact = base_env.sim.data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        if geom1 in base_env.fingertip_geom_ids and geom2 in base_env.can_geom_ids:
            fingers_in_contact.add(geom1)
        elif geom2 in base_env.fingertip_geom_ids and geom1 in base_env.can_geom_ids:
            fingers_in_contact.add(geom2)

    num_fingers = len(fingers_in_contact)
    if num_fingers <= 1:
        grasp_bin = 0  # weak
    elif num_fingers == 2:
        grasp_bin = 1  # stable
    else:
        grasp_bin = 2  # strong

    # Speed bin (angular velocity)
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        base_env.sim.model, base_env.sim.data,
        mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0
    )
    angular_velocity_z = abs(obj_vel[2])

    if angular_velocity_z < 0.02:
        speed_bin = 0  # still
    elif angular_velocity_z < 0.1:
        speed_bin = 1  # rotating
    else:
        speed_bin = 2  # fast

    # State ID: 54 states total (3 grasp x 3 speed x 6 progress)
    state_id = grasp_bin * 18 + speed_bin * 6 + progress_bin

    return state_id, progress_bin, grasp_bin, speed_bin, z_rotation, goal_progress


# ============================
# Training Parameters
# ============================
NUM_STATES = 54   # 3 grasp x 3 speed x 6 progress (was 36)
NUM_ACTIONS = 5   # GRASP, RELEASE, ROT_POS, HOLD, ROT_NEG
LEARNING_RATE = 0.15
DISCOUNT = 0.99
EPSILON = 0.8          # Start with high exploration
EPSILON_DECAY = 0.997  # Slow decay for thorough exploration
MIN_EPSILON = 0.05
NUM_EPISODES = 3000    # More episodes for curriculum learning
MAX_STEPS = 400

# Curriculum: train on progressively harder goals
CURRICULUM = [45, 45, 60, 60, 90, 90]  # Repeat each goal for stability

ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROT_POS",
    3: "HOLD",
    4: "ROT_NEG"
}


# ============================
# Initialize Q-table
# ============================
Q_TABLE_PATH = 'q_table_v2.npy'

if os.path.exists(Q_TABLE_PATH):
    print(f"Loading existing Q-table from {Q_TABLE_PATH}")
    q_table = np.load(Q_TABLE_PATH)
    if q_table.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"Shape mismatch! Reinitializing...")
        q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
else:
    print("Initializing new Q-table with zeros")
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

print(f"Q-table shape: {q_table.shape}")


# ============================
# Initialize Environment
# ============================
initial_goal = CURRICULUM[0]
base_env = CanRotateEnv(target_degrees=initial_goal, render_mode="headless")
env = ActionTranslator(base_env)

print(f"\nStarting curriculum training:")
print(f"  Goals: {CURRICULUM}")
print(f"  Episodes per goal: {NUM_EPISODES // len(CURRICULUM)}")
print(f"  Total episodes: {NUM_EPISODES}")


# ============================
# Training Loop
# ============================
reward_tracker = []
episodes_per_goal = NUM_EPISODES // len(CURRICULUM)

for episode in range(NUM_EPISODES):
    # Curriculum: change goal based on episode
    curriculum_idx = min(episode // episodes_per_goal, len(CURRICULUM) - 1)
    current_goal = CURRICULUM[curriculum_idx]
    base_env.rotation_goal_delta = np.deg2rad(current_goal)

    # Reset environment
    observation, _ = env.reset()
    state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

    total_reward = 0.0
    step = 0

    while step < MAX_STEPS:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Execute action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

        total_reward += reward
        step += 1

        # Q-learning update
        if terminated:
            if goal_prog < np.deg2rad(5):
                best_future_q = np.max(q_table[new_state])  # Success
            else:
                best_future_q = 0  # Dropped cube
        else:
            best_future_q = np.max(q_table[new_state])

        prev_q = q_table[state, action]
        new_q = prev_q + LEARNING_RATE * (reward + DISCOUNT * best_future_q - prev_q)
        q_table[state, action] = new_q

        if terminated:
            break

        state = new_state

    reward_tracker.append(total_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    # Logging
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(reward_tracker[-50:])
        print(f"Ep {episode+1:4d}/{NUM_EPISODES} | Goal: {current_goal:3d}° | "
              f"Avg Reward: {avg_reward:7.2f} | Epsilon: {EPSILON:.3f} | "
              f"Final Z: {np.rad2deg(z_rot):6.1f}°")

    # Checkpoint saves
    if (episode + 1) % 500 == 0:
        np.save(f'q_table_v2_checkpoint_{episode+1}.npy', q_table)
        print(f"  -> Checkpoint saved: q_table_v2_checkpoint_{episode+1}.npy")


# ============================
# Save Final Q-table
# ============================
np.save('q_table_v2_final.npy', q_table)
print(f"\nTraining complete!")
print(f"Final Q-table saved as: q_table_v2_final.npy")

# Print Q-table summary
print(f"\n=== Q-table Summary ===")
print(f"Non-zero entries: {np.count_nonzero(q_table)} / {q_table.size}")
print(f"Max Q-value: {np.max(q_table):.2f}")
print(f"Best actions per state:")
for s in range(NUM_STATES):
    if np.max(q_table[s]) > 0:
        best = np.argmax(q_table[s])
        print(f"  State {s:2d}: {ACTION_NAMES[best]} (Q={q_table[s, best]:.1f})")
