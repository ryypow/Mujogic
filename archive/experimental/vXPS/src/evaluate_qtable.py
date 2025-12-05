"""
Evaluate a trained Q-table on the CanRotateEnv
Usage: python evaluate_qtable.py [--episodes N] [--render] [--goal DEGREES]
"""
import sys
import os
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from inhand_env import CanRotateEnv
from discrete_translator import ActionTranslator

# Action names for display
ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROT_POS",
    3: "HOLD",
    4: "ROT_NEG"
}

def state_translator(observation, env):
    """Convert continuous observation to discrete state (same as training)"""
    base_env = env.unwrapped

    # Get cube rotation
    cube_quat_mujoco = observation[-4:]
    cube_quat_scipy = np.array([cube_quat_mujoco[1], cube_quat_mujoco[2],
                                 cube_quat_mujoco[3], cube_quat_mujoco[0]])
    r = Rotation.from_quat(cube_quat_scipy)
    euler = r.as_euler('xyz')
    z_rotation = euler[2]

    goal = base_env.target_rotation
    goal_progress = goal - z_rotation

    # Direction bin
    direction_bin = 0 if goal_progress >= 0 else 1

    # Progress bin
    goal_progress_abs = abs(goal_progress)
    if goal_progress_abs > np.deg2rad(30):
        progress_bin = 0  # far
    elif goal_progress_abs > np.deg2rad(15):
        progress_bin = 1  # getting there
    elif goal_progress_abs > np.deg2rad(5):
        progress_bin = 2  # acceptable
    else:
        progress_bin = 3  # at goal

    # Grasp bin (fingers in contact)
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
        grasp_bin = 0
    elif num_fingers == 2:
        grasp_bin = 1
    else:
        grasp_bin = 2

    # Speed bin
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(base_env.sim.model, base_env.sim.data,
                              mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0)
    angular_velocity_z = abs(obj_vel[2])

    if angular_velocity_z < 0.02:
        speed_bin = 0
    elif angular_velocity_z < 0.1:
        speed_bin = 1
    else:
        speed_bin = 2

    state_id = direction_bin * 36 + grasp_bin * 12 + speed_bin * 4 + progress_bin

    return state_id, progress_bin, grasp_bin, speed_bin, z_rotation, goal_progress


def evaluate(q_table_path, num_episodes=10, render=True, goal_degrees=45, max_steps=300):
    """Run evaluation episodes using the trained Q-table"""

    # Load Q-table
    q_table = np.load(q_table_path)
    print(f"Loaded Q-table from {q_table_path}")
    print(f"Q-table shape: {q_table.shape}")
    print(f"Q-table stats: min={q_table.min():.2f}, max={q_table.max():.2f}, mean={q_table.mean():.2f}")
    print()

    # Initialize environment
    render_mode = "human" if render else "headless"
    base_env = CanRotateEnv(goal_degrees, render_mode=render_mode)
    env = ActionTranslator(base_env)

    # Tracking metrics
    episode_rewards = []
    episode_steps = []
    successes = 0
    drops = 0

    print(f"Running {num_episodes} evaluation episodes (Goal: {goal_degrees} degrees)")
    print("=" * 70)

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        target_deg = np.rad2deg(env.unwrapped.target_rotation)
        start_deg = np.rad2deg(z_rot)

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Target: {target_deg:.1f}° | Start: {start_deg:.1f}°")
        print("-" * 50)

        while step < max_steps and not terminated and not truncated:
            # Greedy action selection (no exploration)
            action = np.argmax(q_table[state])

            # Execute action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

            total_reward += reward
            step += 1
            state = new_state

            # Print progress every 50 steps
            if step % 50 == 0:
                action_name = ACTION_NAMES[action]
                print(f"  Step {step:3d} | {action_name:8s} | Z: {np.rad2deg(z_rot):6.1f}° | "
                      f"Progress: {np.rad2deg(goal_prog):6.1f}° | Reward: {reward:7.2f}")

        # Episode summary
        final_error = abs(goal_prog)
        success = final_error < np.deg2rad(5)
        dropped = env.unwrapped.sim.data.xpos[env.unwrapped.obj_body_id][2] < \
                  (env.unwrapped.sim.data.site_xpos[env.unwrapped.site_id][2] - 0.05)

        if success:
            successes += 1
            status = "SUCCESS"
        elif dropped:
            drops += 1
            status = "DROPPED"
        else:
            status = "TIMEOUT"

        print(f"  Result: {status} | Steps: {step} | Final Error: {np.rad2deg(final_error):.1f}° | "
              f"Total Reward: {total_reward:.2f}")

        episode_rewards.append(total_reward)
        episode_steps.append(step)

    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes:       {num_episodes}")
    print(f"Success Rate:   {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Drop Rate:      {drops}/{num_episodes} ({100*drops/num_episodes:.1f}%)")
    print(f"Avg Reward:     {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Avg Steps:      {np.mean(episode_steps):.1f}")
    print(f"Best Reward:    {max(episode_rewards):.2f}")
    print(f"Worst Reward:   {min(episode_rewards):.2f}")

    env.close()
    return episode_rewards, successes, drops


def analyze_qtable(q_table_path):
    """Print analysis of the Q-table values"""
    q_table = np.load(q_table_path)

    print("\n" + "=" * 70)
    print("Q-TABLE ANALYSIS")
    print("=" * 70)

    print(f"\nShape: {q_table.shape} (states x actions)")
    print(f"Total entries: {q_table.size}")
    print(f"Non-zero entries: {np.count_nonzero(q_table)} ({100*np.count_nonzero(q_table)/q_table.size:.1f}%)")

    print(f"\nValue statistics:")
    print(f"  Min:  {q_table.min():.4f}")
    print(f"  Max:  {q_table.max():.4f}")
    print(f"  Mean: {q_table.mean():.4f}")
    print(f"  Std:  {q_table.std():.4f}")

    print(f"\nBest action per state:")
    for state in range(min(10, q_table.shape[0])):  # Show first 10 states
        best_action = np.argmax(q_table[state])
        best_value = q_table[state, best_action]
        if best_value != 0:
            print(f"  State {state:2d}: {ACTION_NAMES[best_action]:8s} (Q={best_value:.2f})")

    print(f"\nAction frequency (best action counts):")
    best_actions = np.argmax(q_table, axis=1)
    for action_id, action_name in ACTION_NAMES.items():
        count = np.sum(best_actions == action_id)
        print(f"  {action_name:8s}: {count} states ({100*count/q_table.shape[0]:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Q-table")
    parser.add_argument("--qtable", type=str, default="q_table.npy", help="Path to Q-table file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Show visualization")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--goal", type=int, default=45, help="Goal rotation in degrees")
    parser.add_argument("--analyze", action="store_true", help="Just analyze Q-table, don't run episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")

    args = parser.parse_args()

    # Determine render mode
    render = not args.headless
    if args.render:
        render = True

    # Run analysis
    analyze_qtable(args.qtable)

    # Run evaluation unless --analyze only
    if not args.analyze:
        print()
        evaluate(args.qtable, args.episodes, render, args.goal, args.max_steps)
