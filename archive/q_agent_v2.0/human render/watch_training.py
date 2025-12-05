"""
Watch the trained agent in action with visualization.
Loads the latest Q-table and runs episodes with render_mode='human'.
"""
import os
import numpy as np
from inhand_env import CanRotateEnv
from discrete_translator import ActionTranslator
from q_agent_v2 import state_translator

# Load Q-table (try checkpoint first, then final)
q_table = None
for path in ['q_table_v2_checkpoint_500.npy', 'q_table_v2_final.npy', 'q_table_v2.npy']:
    if os.path.exists(path):
        q_table = np.load(path)
        print(f"Loaded Q-table from: {path}")
        print(f"Shape: {q_table.shape}")
        break

if q_table is None:
    print("No Q-table found! Using random actions.")
    q_table = np.zeros((54, 5))  # Match new state space

# Create environment with visualization
target_goal = 45  # Change to 60 or 90 to test other goals
base_env = CanRotateEnv(target_degrees=target_goal, render_mode='human')
env = ActionTranslator(base_env)

print(f"\nWatching agent attempt {target_goal}° rotation...")
print("Press Ctrl+C to stop\n")

try:
    for episode in range(100):
        obs, _ = env.reset()
        state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(obs, env)

        total_reward = 0
        max_rotation = 0

        for step in range(400):
            # Use learned policy (greedy)
            if state < q_table.shape[0]:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(obs, env)

            total_reward += reward
            current_rotation = np.rad2deg(z_rot)
            max_rotation = max(max_rotation, abs(current_rotation))

            if terminated or truncated:
                break

        final_z = np.rad2deg(z_rot)
        print(f"Ep {episode+1:3d} | Goal: {target_goal}° | Reward: {total_reward:7.1f} | "
              f"Final Z: {final_z:6.1f}° | Max: {max_rotation:6.1f}°")

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    env.close()
