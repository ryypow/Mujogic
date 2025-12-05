"""
Evaluation script for Q-table V2
Tests the trained agent with human rendering to observe behavior.
"""
import sys
import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from discrete_translator import ActionTranslator
from inhand_env import CanRotateEnv


def state_translator(observation, env):
    """Same state translator as q_agent_v2.py"""
    base_env = env.unwrapped

    cube_quat_mujoco = observation[-4:]
    cube_quat_scipy = np.array([
        cube_quat_mujoco[1], cube_quat_mujoco[2],
        cube_quat_mujoco[3], cube_quat_mujoco[0]
    ])
    r = Rotation.from_quat(cube_quat_scipy)
    z_rotation = r.as_euler('xyz')[2]

    goal = base_env.target_rotation
    goal_progress = abs(goal - z_rotation)
    goal_abs = abs(goal)

    if goal_abs > 0:
        progress_ratio = goal_progress / goal_abs
    else:
        progress_ratio = 0

    if progress_ratio > 0.7:
        progress_bin = 0
    elif progress_ratio > 0.3:
        progress_bin = 1
    elif progress_ratio > 0.1:
        progress_bin = 2
    else:
        progress_bin = 3

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

    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        base_env.sim.model, base_env.sim.data,
        mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0
    )
    angular_velocity_z = abs(obj_vel[2])

    if angular_velocity_z < 0.02:
        speed_bin = 0
    elif angular_velocity_z < 0.1:
        speed_bin = 1
    else:
        speed_bin = 2

    state_id = grasp_bin * 12 + speed_bin * 4 + progress_bin
    return state_id, progress_bin, grasp_bin, speed_bin, z_rotation, goal_progress


ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROT_POS",
    3: "HOLD",
    4: "ROT_NEG"
}


def evaluate(q_table_path, goal_degrees, num_episodes=5, render_mode="human"):
    """Run evaluation episodes using trained Q-table."""

    print("=" * 60)
    print(f"EVALUATION: {q_table_path}")
    print(f"Goal: {goal_degrees}° | Episodes: {num_episodes}")
    print("=" * 60)

    # Load Q-table
    if not os.path.exists(q_table_path):
        print(f"ERROR: Q-table not found at {q_table_path}")
        return

    q_table = np.load(q_table_path)
    print(f"Loaded Q-table shape: {q_table.shape}")

    # Initialize environment
    base_env = CanRotateEnv(target_degrees=goal_degrees, render_mode=render_mode)
    env = ActionTranslator(base_env)

    successes = 0
    total_rewards = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        observation, _ = env.reset()
        state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

        target = base_env.target_rotation
        print(f"Target: {np.rad2deg(target):.1f}° | Start Z: {np.rad2deg(z_rot):.1f}°")

        total_reward = 0.0
        step = 0
        max_steps = 400

        while step < max_steps:
            # Greedy action (no exploration)
            action = np.argmax(q_table[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_obs, env)

            total_reward += reward
            step += 1

            # Print every 20 steps
            if step % 20 == 0:
                print(f"  Step {step:3d} | {ACTION_NAMES[action]:7s} | "
                      f"Z: {np.rad2deg(z_rot):6.1f}° | "
                      f"Remaining: {np.rad2deg(goal_prog):5.1f}° | "
                      f"State: {new_state}")

            if terminated:
                if goal_prog < np.deg2rad(5):
                    print(f"  SUCCESS! Reached target at step {step}")
                    successes += 1
                else:
                    print(f"  FAILED: Cube dropped at step {step}")
                break

            if truncated:
                print(f"  TIMEOUT: Max steps reached")
                break

            state = new_state

        total_rewards.append(total_reward)
        print(f"  Total reward: {total_reward:.2f}")

    env.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Best reward: {np.max(total_rewards):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Q-table V2")
    parser.add_argument("--qtable", type=str, default="q_table_v2_final.npy",
                        help="Path to Q-table file")
    parser.add_argument("--goal", type=int, default=90,
                        help="Target rotation in degrees")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--headless", action="store_true",
                        help="Run without rendering")

    args = parser.parse_args()

    render = "headless" if args.headless else "human"
    evaluate(args.qtable, args.goal, args.episodes, render)
