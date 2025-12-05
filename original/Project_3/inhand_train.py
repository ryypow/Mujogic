# inhand_train.py - Training loop for Q-Learning cube rotation
import os
import numpy as np
from inhand_env import CanRotateEnv
from MinimalTranslator import MinimalTranslator
from RLagent import QLearningAgent

# Create output directory
log_dir = "training_output/"
os.makedirs(log_dir, exist_ok=True)

# --- Configuration ---
NUM_EPISODES = 2000
MAX_STEPS = 300
CHECKPOINT_INTERVAL = 500

# --- Initialize Environment ---
print("Initializing environment...")
base_env = CanRotateEnv(render_mode="headless")
env = MinimalTranslator(base_env)

# --- Initialize Agent ---
print("Initializing Q-Learning agent...")
agent = QLearningAgent(
    num_states=18,       # 3 speed x 6 progress
    num_actions=8,       # HOLD, THUMB x2, FINGER3 x3, FINGER2 x2
    learning_rate=0.1,
    discount=0.99,
    epsilon=0.8,
    epsilon_decay=0.997,
    min_epsilon=0.05
)

# Optional: load existing Q-table to continue training
# agent.load("training_output/q_table_final.npy")

print(f"\nStarting training for {NUM_EPISODES} episodes...")
print(f"State space: {agent.num_states} states")
print(f"Action space: {agent.num_actions} actions")
print(f"  0: HOLD")
print(f"  1: THUMB_PUSH")
print(f"  2: THUMB_RETRACT")
print(f"  3: FINGER3_CURL")
print(f"  4: FINGER3_NUDGE")
print(f"  5: FINGER3_RETRACT")
print(f"  6: FINGER2_PUSH")
print(f"  7: FINGER2_RETRACT")

# --- Training Loop ---
reward_history = []

for episode in range(NUM_EPISODES):
    # Reset environment
    obs, _ = env.reset()
    state, z_rot, goal_progress = agent.get_state(env)

    total_reward = 0.0
    step = 0

    while step < MAX_STEPS:
        # Get action from agent
        action = agent.get_action(state)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state, z_rot, goal_progress = agent.get_state(env)

        # Learn from transition
        done = terminated or truncated
        agent.learn(state, action, reward, next_state, done)

        total_reward += reward
        step += 1
        state = next_state

        if done:
            break

    # End of episode
    reward_history.append(total_reward)
    agent.decay_epsilon()

    # Logging every 50 episodes
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(reward_history[-50:])
        print(f"Episode {episode+1:4d}/{NUM_EPISODES} | "
              f"Avg Reward: {avg_reward:7.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Final Z: {z_rot:6.1f}°")

    # Checkpoint saves
    if (episode + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = f"{log_dir}q_table_checkpoint_{episode+1}.npy"
        agent.save(checkpoint_path)

# --- Save final model ---
agent.save(f"{log_dir}q_table_final.npy")

# --- Cleanup ---
env.close()
print("\nTraining complete!")
print(f"Final Q-table saved to {log_dir}q_table_final.npy")
