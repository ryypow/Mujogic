"""
Evaluate a trained Q-table on the CanRotateEnv
Based on q_agent_finetune.py state translator (288 states)

Usage:
    python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy
    python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy --episodes 20 --headless
    python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy --analyze
"""
import sys
import os
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inhand_env import CanRotateEnv
from discrete_translator import ActionTranslator


# Constants matching q_agent_finetune.py
NUM_STATES = 288  # 2 rotation × 4 goal × 3 grasp × 3 speed × 4 progress
NUM_ACTIONS = 5
GOAL_OPTIONS = [45, 90, 180, 270, 360]

ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROT_POS",
    3: "HOLD",
    4: "ROT_NEG"
}


def state_translator(observation, env):
    """
    Convert continuous observation to discrete state.
    EXACT COPY from q_agent_finetune.py to ensure state mapping matches training.

    State space: 288 states
    - rotation_bin: 2 (positive/negative z-rotation)
    - goal_bin: 4 (goal magnitude bins)
    - grasp_bin: 3 (weak/stable/strong)
    - speed_bin: 3 (stopped/rotating/fast)
    - progress_bin: 4 (far/making progress/close/at goal)
    """
    base_env = env.unwrapped

    # ==== Get cube rotation from quaternion
    # MuJoCo quaternion format: [w, x, y, z], SciPy expects: [x, y, z, w]
    cube_quat_mujoco = observation[-4:]  # [qw, qx, qy, qz]
    cube_quat_scipy = np.array([cube_quat_mujoco[1], cube_quat_mujoco[2],
                                 cube_quat_mujoco[3], cube_quat_mujoco[0]])
    r = Rotation.from_quat(cube_quat_scipy)
    euler = r.as_euler('xyz')
    z_rotation = euler[2]

    goal = base_env.target_rotation
    goal_progress = goal - z_rotation

    # ==== Grasp bin - fingers in contact
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
    else:  # >= 3
        grasp_bin = 2  # strong

    # ==== Speed bin - angular velocity
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(base_env.sim.model, base_env.sim.data,
                              mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0)
    angular_velocity_z = abs(obj_vel[2])

    if angular_velocity_z < 0.02:
        speed_bin = 0  # not spinning
    elif angular_velocity_z < 0.1:
        speed_bin = 1  # rotating
    else:
        speed_bin = 2  # rotating fast

    # ==== Goal magnitude bin - aligned with GOAL = [45, 90, 180, 270, 360]
    goal_abs = abs(goal)
    if goal_abs > np.deg2rad(180):
        goal_bin = 3  # 181-360°
    elif goal_abs > np.deg2rad(90):
        goal_bin = 2  # 91-180°
    elif goal_abs > np.deg2rad(45):
        goal_bin = 1  # 46-90°
    else:
        goal_bin = 0  # 0-45°

    # ==== Relative progress bin
    if goal_abs > 0:
        progress_ratio = abs(goal_progress) / goal_abs
    else:
        progress_ratio = 0

    if progress_ratio > 0.7:
        progress_bin = 0  # far
    elif progress_ratio > 0.3:
        progress_bin = 1  # making progress
    elif progress_ratio > 0.1:
        progress_bin = 2  # close
    else:
        progress_bin = 3  # at goal

    # ==== Rotation bin
    rotation_bin = 0 if z_rotation >= 0 else 1

    # ==== Compute state ID (must match training exactly!)
    state_id = rotation_bin * 144 + goal_bin * 36 + grasp_bin * 12 + speed_bin * 4 + progress_bin

    return state_id, {
        'rotation_bin': rotation_bin,
        'goal_bin': goal_bin,
        'grasp_bin': grasp_bin,
        'speed_bin': speed_bin,
        'progress_bin': progress_bin,
        'z_rotation': z_rotation,
        'goal_progress': goal_progress,
        'num_fingers': num_fingers,
        'angular_velocity': angular_velocity_z
    }


def evaluate(q_table_path, num_episodes=10, render=True, goal_degrees=45, max_steps=400):
    """Run evaluation episodes using the trained Q-table"""

    # Load Q-table
    q_table = np.load(q_table_path)
    print(f"Loaded Q-table from {q_table_path}")
    print(f"Q-table shape: {q_table.shape}")

    # Validate shape
    if q_table.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"ERROR: Q-table shape mismatch!")
        print(f"  Expected: ({NUM_STATES}, {NUM_ACTIONS})")
        print(f"  Got: {q_table.shape}")
        return None, 0, 0

    print(f"Q-table stats: min={q_table.min():.2f}, max={q_table.max():.2f}, mean={q_table.mean():.2f}")
    print()

    # Initialize environment
    render_mode = "human" if render else "headless"
    base_env = CanRotateEnv(goal_degrees, render_mode=render_mode)
    env = ActionTranslator(base_env)

    # Tracking metrics
    episode_rewards = []
    episode_steps = []
    final_errors = []
    successes = 0
    drops = 0

    print(f"Running {num_episodes} evaluation episodes (Goal: {goal_degrees}°)")
    print("=" * 80)

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state, state_info = state_translator(observation, env)

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        target_deg = np.rad2deg(env.unwrapped.target_rotation)
        start_deg = np.rad2deg(state_info['z_rotation'])

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Target: {target_deg:.1f}° | Start: {start_deg:.1f}° | Goal Delta: {goal_degrees}°")
        print("-" * 60)

        action_counts = {i: 0 for i in range(NUM_ACTIONS)}

        while step < max_steps and not terminated and not truncated:
            # Greedy action selection (no exploration)
            action = np.argmax(q_table[state])
            action_counts[action] += 1

            # Execute action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            new_state, state_info = state_translator(next_observation, env)

            total_reward += reward
            step += 1
            state = new_state

            # Print progress every 50 steps
            if step % 50 == 0 or terminated:
                z_rot = state_info['z_rotation']
                goal_prog = state_info['goal_progress']
                action_name = ACTION_NAMES[action]
                print(f"  Step {step:3d} | {action_name:8s} | Z: {np.rad2deg(z_rot):7.1f}° | "
                      f"Remaining: {np.rad2deg(goal_prog):7.1f}° | Fingers: {state_info['num_fingers']} | "
                      f"Reward: {reward:7.2f}")

        # Episode summary
        final_error = abs(state_info['goal_progress'])
        final_errors.append(np.rad2deg(final_error))

        success = final_error < np.deg2rad(5)
        dropped = env.unwrapped.sim.data.xpos[env.unwrapped.obj_body_id][2] < \
                  (env.unwrapped.sim.data.site_xpos[env.unwrapped.site_id][2] - 0.05)

        if success:
            successes += 1
            status = "SUCCESS"
        elif dropped:
            drops += 1
            status = "DROPPED"
        elif truncated:
            status = "TIMEOUT"
        else:
            status = "TERMINATED"

        print(f"\n  Result: {status}")
        print(f"  Steps: {step} | Final Error: {np.rad2deg(final_error):.1f}° | Total Reward: {total_reward:.2f}")
        print(f"  Actions: " + " | ".join([f"{ACTION_NAMES[k]}: {v}" for k, v in action_counts.items()]))

        episode_rewards.append(total_reward)
        episode_steps.append(step)

    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Q-table:        {q_table_path}")
    print(f"Goal:           {goal_degrees}°")
    print(f"Episodes:       {num_episodes}")
    print(f"Max Steps:      {max_steps}")
    print("-" * 40)
    print(f"Success Rate:   {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Drop Rate:      {drops}/{num_episodes} ({100*drops/num_episodes:.1f}%)")
    print(f"Timeout Rate:   {num_episodes - successes - drops}/{num_episodes} ({100*(num_episodes-successes-drops)/num_episodes:.1f}%)")
    print("-" * 40)
    print(f"Avg Reward:     {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print(f"Avg Steps:      {np.mean(episode_steps):.1f}")
    print(f"Avg Final Error:{np.mean(final_errors):.1f}° (+/- {np.std(final_errors):.1f}°)")
    print(f"Best Reward:    {max(episode_rewards):.2f}")
    print(f"Worst Reward:   {min(episode_rewards):.2f}")
    print(f"Min Final Error:{min(final_errors):.1f}°")

    env.close()
    return episode_rewards, successes, drops


def analyze_qtable(q_table_path):
    """Print detailed analysis of the Q-table values"""
    q_table = np.load(q_table_path)

    print("\n" + "=" * 80)
    print("Q-TABLE ANALYSIS")
    print("=" * 80)

    print(f"\nFile: {q_table_path}")
    print(f"Shape: {q_table.shape} (states × actions)")

    if q_table.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"\nWARNING: Shape mismatch with training!")
        print(f"  Expected: ({NUM_STATES}, {NUM_ACTIONS})")
        return

    print(f"Total entries: {q_table.size}")
    non_zero = np.count_nonzero(q_table)
    print(f"Non-zero entries: {non_zero} ({100*non_zero/q_table.size:.1f}%)")
    print(f"Visited states: {np.sum(np.any(q_table != 0, axis=1))} / {NUM_STATES}")

    print(f"\nValue Statistics:")
    print(f"  Min:    {q_table.min():.4f}")
    print(f"  Max:    {q_table.max():.4f}")
    print(f"  Mean:   {q_table.mean():.4f}")
    print(f"  Std:    {q_table.std():.4f}")
    print(f"  Median: {np.median(q_table):.4f}")

    print(f"\nAction Distribution (best action per state):")
    best_actions = np.argmax(q_table, axis=1)
    for action_id, action_name in ACTION_NAMES.items():
        count = np.sum(best_actions == action_id)
        print(f"  {action_name:8s}: {count:3d} states ({100*count/NUM_STATES:5.1f}%)")

    print(f"\nState Breakdown (rotation_bin × goal_bin × grasp_bin × speed_bin × progress_bin):")
    print(f"  rotation_bin: 2 bins (0=positive, 1=negative)")
    print(f"  goal_bin:     4 bins (0=0-45°, 1=46-90°, 2=91-180°, 3=181-360°)")
    print(f"  grasp_bin:    3 bins (0=weak, 1=stable, 2=strong)")
    print(f"  speed_bin:    3 bins (0=stopped, 1=rotating, 2=fast)")
    print(f"  progress_bin: 4 bins (0=far, 1=progress, 2=close, 3=goal)")

    # Show Q-values for key states
    print(f"\nSample Q-values for key states:")
    print(f"  {'State':>6} | {'Rot':>3} | {'Goal':>4} | {'Grasp':>5} | {'Speed':>5} | {'Prog':>4} | Best Action | Q-values")
    print("  " + "-" * 90)

    sample_states = [0, 1, 2, 3, 36, 72, 144, 180, 216, 252]  # Sample across state space
    for state in sample_states:
        if state < NUM_STATES:
            # Decode state
            rotation_bin = state // 144
            remainder = state % 144
            goal_bin = remainder // 36
            remainder = remainder % 36
            grasp_bin = remainder // 12
            remainder = remainder % 12
            speed_bin = remainder // 4
            progress_bin = remainder % 4

            best_action = np.argmax(q_table[state])
            best_value = q_table[state, best_action]
            q_vals = ", ".join([f"{v:6.2f}" for v in q_table[state]])

            if np.any(q_table[state] != 0):
                print(f"  {state:6d} | {rotation_bin:3d} | {goal_bin:4d} | {grasp_bin:5d} | {speed_bin:5d} | {progress_bin:4d} | "
                      f"{ACTION_NAMES[best_action]:8s} | [{q_vals}]")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Q-table (matches q_agent_finetune.py state space)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy
  python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy --episodes 20 --goal 90
  python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy --headless --episodes 50
  python tests/evaluate_qtable_v2.py --qtable q_table_checkpoint_1400.npy --analyze
        """
    )
    parser.add_argument("--qtable", type=str, default="q_table_checkpoint_1400.npy",
                        help="Path to Q-table file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                        help="Show visualization (default)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--goal", type=int, default=45,
                        help="Goal rotation in degrees")
    parser.add_argument("--analyze", action="store_true",
                        help="Only analyze Q-table, don't run episodes")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps per episode")

    args = parser.parse_args()

    # Validate Q-table path
    if not os.path.exists(args.qtable):
        # Try looking in parent directory
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.qtable)
        if os.path.exists(parent_path):
            args.qtable = parent_path
        else:
            print(f"ERROR: Q-table file not found: {args.qtable}")
            print(f"  Also checked: {parent_path}")
            sys.exit(1)

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


if __name__ == "__main__":
    main()
