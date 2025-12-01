"""
Evaluate trained Q-learning agent for cube rotation
Loads q_table.npy and runs episodes with greedy policy (no exploration)
"""
import sys
import os
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from discrete_translator import ActionTranslator
from inhand_env import CanRotateEnv

# Import state translator from training script
from q_agent import state_translator, ACTION_NAMES, NUM_STATES, NUM_ACTIONS


def evaluate(q_table_path, goal_degrees, num_episodes=10, render_mode="human", max_steps=300):
    """
    Evaluate trained agent with greedy policy.

    Args:
        q_table_path: Path to saved q_table.npy
        goal_degrees: Target rotation in degrees
        num_episodes: Number of evaluation episodes
        render_mode: "human" for visualization, "headless" for fast evaluation
        max_steps: Maximum steps per episode
    """
    # Load trained Q-table
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table not found at {q_table_path}")
        print("Train the agent first by running: python q_agent.py")
        sys.exit(1)

    q_table = np.load(q_table_path)
    print(f"Loaded Q-table from {q_table_path}")
    print(f"Q-table shape: {q_table.shape}")

    # Verify Q-table dimensions
    if q_table.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"Warning: Q-table shape {q_table.shape} doesn't match expected ({NUM_STATES}, {NUM_ACTIONS})")

    # Initialize environment
    base_env = CanRotateEnv(goal_degrees, render_mode=render_mode)
    env = ActionTranslator(base_env)

    print(f"\nEvaluating agent for {num_episodes} episodes")
    print(f"Target rotation: {goal_degrees} degrees")
    print(f"Render mode: {render_mode}")
    print("=" * 70)

    # Tracking metrics
    episode_rewards = []
    episode_steps = []
    successes = 0

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)

        total_reward = 0.0
        step = 0
        done = False

        base = env.unwrapped
        print(f"\n=== Episode {episode + 1}/{num_episodes} | Target: {np.rad2deg(base.target_rotation):.1f}° | Start Z-Rot: {np.rad2deg(z_rot):.1f}° ===")

        while step < max_steps and not done:
            # Greedy action selection (no exploration)
            action = np.argmax(q_table[state])

            # Execute action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

            total_reward += reward
            step += 1
            done = terminated or truncated

            # Print progress every 20 steps
            if step % 20 == 0 or done:
                action_name = ACTION_NAMES.get(int(action), "???")
                print(f"Step {step:4d} | {action_name:8s} | State {new_state:3d} | Z-Rot: {np.rad2deg(z_rot):6.1f}° | G:{grasp_bin} S:{speed_bin} P:{progress_bin} | Reward: {reward:7.2f}")

            state = new_state

        # Check if goal was achieved
        goal_achieved = abs(goal_prog) < np.deg2rad(5)
        if goal_achieved:
            successes += 1
            print(f"*** SUCCESS! Goal achieved in {step} steps ***")
        elif terminated:
            print(f"*** TERMINATED (cube dropped?) at step {step} ***")
        else:
            print(f"*** TRUNCATED (max steps) at step {step} ***")

        episode_rewards.append(total_reward)
        episode_steps.append(step)
        print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes}/{num_episodes} ({100 * successes / num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Average Steps: {np.mean(episode_steps):.1f} (+/- {np.std(episode_steps):.1f})")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")

    env.close()

    return {
        "success_rate": successes / num_episodes,
        "avg_reward": np.mean(episode_rewards),
        "avg_steps": np.mean(episode_steps),
        "rewards": episode_rewards,
        "steps": episode_steps
    }


def print_q_table_analysis(q_table_path):
    """Analyze and print Q-table statistics."""
    q_table = np.load(q_table_path)

    print("\n" + "=" * 70)
    print("Q-TABLE ANALYSIS")
    print("=" * 70)

    # Overall statistics
    print(f"Shape: {q_table.shape}")
    print(f"Min Q-value: {q_table.min():.4f}")
    print(f"Max Q-value: {q_table.max():.4f}")
    print(f"Mean Q-value: {q_table.mean():.4f}")
    print(f"Non-zero entries: {np.count_nonzero(q_table)} / {q_table.size} ({100 * np.count_nonzero(q_table) / q_table.size:.1f}%)")

    # Best action per state
    print("\nPreferred actions by state (showing states with learning):")
    print("-" * 50)

    learned_states = np.where(np.any(q_table != 0, axis=1))[0]
    for state in learned_states[:20]:  # Show first 20 learned states
        best_action = np.argmax(q_table[state])
        q_values = q_table[state]

        # Decode state
        direction_bin = state // 36
        remainder = state % 36
        grasp_bin = remainder // 12
        remainder2 = remainder % 12
        speed_bin = remainder2 // 4
        progress_bin = remainder2 % 4

        direction_str = "+Z" if direction_bin == 0 else "-Z"
        grasp_str = ["weak", "stable", "strong"][grasp_bin]
        speed_str = ["still", "slow", "fast"][speed_bin]
        progress_str = ["far", "mid", "close", "goal"][progress_bin]

        print(f"State {state:3d} ({direction_str}, {grasp_str}, {speed_str}, {progress_str}): "
              f"Best={ACTION_NAMES[best_action]:8s} Q={q_values[best_action]:7.2f}")

    if len(learned_states) > 20:
        print(f"... and {len(learned_states) - 20} more learned states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Q-learning agent")
    parser.add_argument("--q-table", type=str, default="q_table_36states.npy", help="Path to Q-table file")
    parser.add_argument("--goal", type=int, default=45, help="Target rotation in degrees")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", type=str, default="human", choices=["human", "headless"],
                        help="Render mode: human (visualize) or headless (fast)")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--analyze", action="store_true", help="Print Q-table analysis")

    args = parser.parse_args()

    if args.analyze:
        print_q_table_analysis(args.q_table)

    evaluate(
        q_table_path=args.q_table,
        goal_degrees=args.goal,
        num_episodes=args.episodes,
        render_mode=args.render,
        max_steps=args.max_steps
    )
